import os, numpy as np, pandas as pd, tensorflow as tf, cv2, time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

# ---- Paths (updated to /content/data) ----
DATA_DIR = "/content/data/ISIC2018_Task3_Training_Input"
LABELS_CSV = "/content/data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
OUT_DIR = "/content/ISIC_Phase2" # Output directory remains in /content
os.makedirs(OUT_DIR, exist_ok=True)
SYN_DIR = os.path.join(OUT_DIR, "synthetic_cgan")
os.makedirs(SYN_DIR, exist_ok=True)

IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS_CGAN = 100    # increase for better quality
EPOCHS_CLS  = 20

# =============== DATA LOADING ===============
df = pd.read_csv(LABELS_CSV)

mel_ids = df[df['MEL'] == 1]['image'].values
nonmel_ids = df[df['MEL'] == 0]['image'].values

def load_images_by_ids(ids, directory, img_size=64, limit=None):
    images, keep_ids = [], []
    count = 0
    for iid in tqdm(ids, desc="Loading"):
        if limit and count >= limit: break
        p = os.path.join(directory, f"{iid}.jpg")
        if os.path.exists(p):
            img = Image.open(p).convert("RGB").resize((img_size, img_size))
            img = np.asarray(img).astype('float32') / 255.0
            images.append(img)
            keep_ids.append(iid)
            count += 1
    return np.array(images), np.array(keep_ids)

# Sample a subset for faster iteration (increase later)
MEL_LIMIT     = 2000
NONMEL_LIMIT  = 2000

real_mel_imgs, real_mel_ids = load_images_by_ids(mel_ids, DATA_DIR, IMG_SIZE, limit=MEL_LIMIT)
nonmel_imgs, nonmel_ids     = load_images_by_ids(nonmel_ids, DATA_DIR, IMG_SIZE, limit=NONMEL_LIMIT)
print("Real melanoma:", real_mel_imgs.shape, "| Non-melanoma:", nonmel_imgs.shape)

# =============== SIMULATE RISK FOR MELANOMA ===============
# UV: mix of low/med/high; Genetic: younger bias (more high)
def sample_uv(n):
    # mixture: 30% low, 40% medium, 30% high
    u = np.random.rand(n)
    out = np.empty(n)
    low_idx  = u < 0.30
    med_idx  = (u >= 0.30) & (u < 0.70)
    high_idx = u >= 0.70
    out[low_idx]  = np.random.uniform(0.1, 0.4, low_idx.sum())
    out[med_idx]  = np.random.uniform(0.4, 0.7, med_idx.sum())
    out[high_idx] = np.random.uniform(0.7, 1.0, high_idx.sum())
    return out

def sample_genetic(n):
    # skewed a bit higher (younger -> higher)
    # use Beta(3,2) scaled to [0.2, 1.0]
    return 0.2 + 0.8*np.random.beta(3, 2, size=n)

mel_uv  = sample_uv(len(real_mel_ids))
mel_gen = sample_genetic(len(real_mel_ids))
mel_risk = np.stack([mel_uv, mel_gen], axis=1)  # shape (N,2)

# For non-melanoma: use zeros (no risk info)
nonmel_risk = np.zeros((len(nonmel_ids), 2), dtype=np.float32)

# Save mapping (optional for reproducibility)
risk_map = pd.DataFrame({
    "image": real_mel_ids,
    "UV_risk": mel_uv,
    "Genetic_risk": mel_gen
})
risk_map.to_csv(os.path.join(OUT_DIR, "real_melanoma_risk_map.csv"), index=False)
print("Saved real melanoma risk map.")


# =============== CONDITIONAL GAN ===============
from tensorflow.keras import layers, Model

def build_generator(latent_dim=100, cond_dim=2):
    z_in   = layers.Input(shape=(latent_dim,))
    c_in   = layers.Input(shape=(cond_dim,))
    x = layers.Concatenate()([z_in, c_in])  # [z, risk]
    x = layers.Dense(8*8*256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8,8,256))(x)

    x = layers.Conv2DTranspose(128, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(3, 5, strides=2, padding="same", use_bias=False, activation="tanh")(x)
    return Model([z_in, c_in], x, name="cGAN_G")

def build_discriminator(cond_dim=2):
    img_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    c_in   = layers.Input(shape=(cond_dim,))

    x = layers.Conv2D(64, 5, strides=2, padding="same")(img_in)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    # concatenate risk conditioning late
    x = layers.Concatenate()([x, c_in])
    x = layers.Dense(1)(x)  # from_logits
    return Model([img_in, c_in], x, name="cGAN_D")

generator = build_generator(LATENT_DIM, 2)
discriminator = build_discriminator(2)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def cgan_train_step(real_batch, cond_batch):
    bs = tf.shape(real_batch)[0]
    noise = tf.random.normal([bs, LATENT_DIM])

    # Generate
    fake_imgs = generator([noise, cond_batch], training=True)

    with tf.GradientTape() as d_tape:
        real_logits = discriminator([real_batch*2.0-1.0, cond_batch], training=True)  # scale to [-1,1]
        fake_logits = discriminator([fake_imgs, cond_batch], training=True)
        d_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # Train G
    noise2 = tf.random.normal([bs, LATENT_DIM])
    with tf.GradientTape() as g_tape:
        gen_imgs = generator([noise2, cond_batch], training=True)
        fake_logits2 = discriminator([gen_imgs, cond_batch], training=True)
        g_loss = bce(tf.ones_like(fake_logits2), fake_logits2)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss

# Dataset for cGAN: melanoma images + their risk
mel_ds = tf.data.Dataset.from_tensor_slices((real_mel_imgs, mel_risk)) \
        .shuffle(len(real_mel_imgs)).batch(BATCH_SIZE).prefetch(2)

def train_cgan(epochs=EPOCHS_CGAN, sample_every=10):
    print("Training cGAN...")
    for ep in range(1, epochs+1):
        d_losses, g_losses = [], []
        for real_b, cond_b in mel_ds:
            d_loss, g_loss = cgan_train_step(real_b, cond_b)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
        if ep % 5 == 0:
            print(f"Epoch {ep}/{epochs} | D: {np.mean(d_losses):.4f} | G: {np.mean(g_losses):.4f}")

        if ep % sample_every == 0:
            # produce a small grid for fixed risks
            grid_risks = np.array([
                [0.2, 0.2], [0.2, 0.8],
                [0.8, 0.2], [0.8, 0.8]
            ], dtype=np.float32)
            noise = np.random.normal(0,1,(len(grid_risks), LATENT_DIM)).astype(np.float32)
            g = generator.predict([noise, grid_risks], verbose=0)
            g = (g+1.0)/2.0
            fig, axs = plt.subplots(1,4, figsize=(8,2))
            for i in range(4):
                axs[i].imshow(np.clip(g[i],0,1))
                axs[i].set_title(f"UV:{grid_risks[i,0]:.1f} GEN:{grid_risks[i,1]:.1f}")
                axs[i].axis('off')
            plt.tight_layout()
            plt.show()

train_cgan()
# Save generator
generator.save(os.path.join(OUT_DIR, "cgan_generator.h5"))
print("✅ cGAN generator saved.")

# =============== SYNTHETIC GENERATION ===============
N_SYN = 1000

syn_uv  = sample_uv(N_SYN)
syn_gen = sample_genetic(N_SYN)
syn_risk = np.stack([syn_uv, syn_gen], axis=1).astype(np.float32)

all_noise = np.random.normal(0,1,(N_SYN, LATENT_DIM)).astype(np.float32)
syn_imgs = generator.predict([all_noise, syn_risk], batch_size=64, verbose=1)  # tanh [-1,1]
syn_imgs = ((syn_imgs + 1.0)/2.0).clip(0,1)

syn_index = []
for i in range(N_SYN):
    fn = f"syn_{i:05d}.png"
    Image.fromarray((syn_imgs[i]*255).astype(np.uint8)).save(os.path.join(SYN_DIR, fn))
    syn_index.append((fn, syn_risk[i,0], syn_risk[i,1]))
syn_df = pd.DataFrame(syn_index, columns=["filename", "UV_risk", "Genetic_risk"])
syn_df.to_csv(os.path.join(OUT_DIR, "synthetic_index.csv"), index=False)
print("✅ Synthetic images + risk saved.")

# =============== BUILD FINAL DATASET FOR CLASSIFIER ===============
# Prepare: real melanoma, synthetic melanoma, non-melanoma
# Labels: melanoma -> 1 for mel, 1 for syn; 0 for non-mel
# Risk: real mel -> mel_risk; syn -> syn_risk; non-mel -> [0,0]

# Load synthetic images back (to avoid GPU mem pressure)
def load_pngs(folder, df, img_size=64):
    arr = []
    for fname in tqdm(df['filename'], desc="Load Synthetic"):
        p = os.path.join(folder, fname)
        img = Image.open(p).convert("RGB").resize((img_size,img_size))
        arr.append(np.asarray(img)/255.0)
    return np.array(arr, dtype=np.float32)

syn_df = pd.read_csv(os.path.join(OUT_DIR, "synthetic_index.csv"))
syn_imgs_np = load_pngs(SYN_DIR, syn_df, IMG_SIZE)
syn_risk_np = syn_df[["UV_risk","Genetic_risk"]].values.astype(np.float32)

# Build final arrays
X_imgs = np.concatenate([real_mel_imgs, syn_imgs_np, nonmel_imgs], axis=0)
R_risk = np.concatenate([mel_risk, syn_risk_np, nonmel_risk], axis=0)
y_mel  = np.concatenate([np.ones(len(real_mel_imgs)), np.ones(len(syn_imgs_np)), np.zeros(len(nonmel_imgs))], axis=0)

# Targets for UV/GEN: only meaningful for melanoma; for non-mel set 0 (or mask in loss)
y_uv  = np.concatenate([mel_risk[:,0], syn_risk_np[:,0], np.zeros(len(nonmel_imgs))], axis=0)
y_gen = np.concatenate([mel_risk[:,1], syn_risk_np[:,1], np.zeros(len(nonmel_imgs))], axis=0)

print("Final dataset:", X_imgs.shape, R_risk.shape, y_mel.shape)

from sklearn.model_selection import train_test_split
Xtr, Xval, Rtr, Rval, ymel_tr, ymel_val, yuv_tr, yuv_val, ygen_tr, ygen_val = train_test_split(
    X_imgs, R_risk, y_mel, y_uv, y_gen, test_size=0.2, stratify=y_mel, random_state=42
)

# =============== MODEL: IMAGE + RISK (3 outputs) ===============
from tensorflow.keras import Input, Model
from tensorflow.keras import layers

def build_risk_cnn():
    img_in = Input(shape=(IMG_SIZE,IMG_SIZE,3), name="image")
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(img_in)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    risk_in = Input(shape=(2,), name="risk")
    r = layers.Dense(16, activation='relu')(risk_in)

    h = layers.Concatenate()([x, r])
    h = layers.Dense(128, activation='relu')(h)
    h = layers.Dropout(0.5)(h)

    out_mel = layers.Dense(1, activation='sigmoid', name="mel_out")(h)
    out_uv  = layers.Dense(1, activation='sigmoid', name="uv_out")(h)
    out_gen = layers.Dense(1, activation='sigmoid', name="gen_out")(h)

    model = Model(inputs=[img_in, risk_in], outputs=[out_mel, out_uv, out_gen])
    model.compile(
        optimizer='adam',
        loss={
            "mel_out": "binary_crossentropy",
            "uv_out": "mse",
            "gen_out": "mse"
        },
        loss_weights={"mel_out":1.0, "uv_out":0.5, "gen_out":0.5},
        metrics={"mel_out":["accuracy"]}
    )
    return model

risk_cnn = build_risk_cnn()
hist = risk_cnn.fit(
    {"image":Xtr, "risk":Rtr},
    {"mel_out": ymel_tr, "uv_out": yuv_tr, "gen_out": ygen_tr},
    validation_data=(
        {"image":Xval, "risk":Rval},
        {"mel_out": ymel_val, "uv_out": yuv_val, "gen_out": ygen_val}
    ),
    epochs=EPOCHS_CLS,
    batch_size=32
)

# --- Grad-CAM function for multi-output model ---
def grad_cam(model, img_array, risk_array, target_output="mel_out"):
    """
    Generate Grad-CAM heatmap for chosen output of multi-output model.
    target_output: "mel_out", "uv_out", or "gen_out"
    """

    # 🔍 Find last Conv2D layer automatically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found in the model!")

    # Build grad_model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.outputs]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model([img_array[None], risk_array[None]])

        # Pick the correct output branch
        if target_output == "mel_out":
            loss = preds[0][:, 0]  # melanoma prob
        elif target_output == "uv_out":
            loss = preds[1][:, 0]  # UV prob
        elif target_output == "gen_out":
            loss = preds[2][:, 0]  # Genetic prob
        else:
            raise ValueError("Invalid target_output. Use 'mel_out', 'uv_out', or 'gen_out'.")

    # Gradient of loss wrt conv feature maps
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_out), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


# --- Overlay visualization ---
def overlay_heatmap(img, heatmap, alpha=0.4, cmap=cv2.COLORMAP_JET):
    """
    Superimpose heatmap on original image.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cmap)
    overlay = cv2.addWeighted(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                              1-alpha, heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- Risk Contribution ---
def risk_contribution(model, img, risk_vec):
    with tf.GradientTape() as tape:
        tape.watch(risk_vec)
        pred = model([img[None], risk_vec[None]], training=False)[0]  # mel_out
        y = pred[:,0]
    grads = tape.gradient(y, risk_vec)[0].numpy()  # shape (2,)
    contrib = grads * risk_vec.numpy()
    contrib = np.maximum(contrib, 0)
    if contrib.sum() > 0:
        contrib = contrib / contrib.sum()
    return contrib, y.numpy()[0]

# --- Final Model Evaluation ---
from sklearn.metrics import classification_report, confusion_matrix

print("--- Final Model Evaluation ---")

# Predict using the trained multi-task model on the validation data
predictions = risk_cnn.predict({"image": Xval, "risk": Rval})
mel_pred_probs = predictions[0] # The first output is for melanoma
y_pred = (mel_pred_probs > 0.5).astype("int32")

# Generate and print the Classification Report
print("Classification Report:")
print(classification_report(ymel_val, y_pred, target_names=['Non-Melanoma', 'Melanoma']))

# Generate and display the Confusion Matrix
cm = confusion_matrix(ymel_val, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Melanoma','Melanoma'],
            yticklabels=['Non-Melanoma','Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Model Confusion Matrix')
plt.show()