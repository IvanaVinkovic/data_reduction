import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, redirect, url_for
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import base64

app = Flask(__name__)

class MDSNet(tf.keras.Model):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MDSNet, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_dims[0], activation='elu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(hidden_dims[1], activation='elu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        ]
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=None)
    
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def pairwise_distances(x):
    dot_product = tf.matmul(x, tf.transpose(x))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.sqrt(tf.maximum(distances, 1e-9)) 
    return distances

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    if request.method == 'POST':
        file = request.files['csv-file']
        dim = request.form['dim']
        method = request.form['method']

        if not file:
            error_message = "Please upload CSV file."
            return render_template('index.html', error_message=error_message)

        df = pd.read_csv(file)

        # Zadrži samo numeričke stupce
        data = df.select_dtypes(include=[np.number])

        # Provjera broja stupaca
        if data.shape[1] < (2 if dim == '2d' else 3):
            error_message = "Number of components cannot be greater than the number of table columns"
            return render_template('index.html', error_message=error_message)

        # Popunjavanje NaN vrijednosti s prosječnom vrijednošću stupca
        imputer = SimpleImputer(strategy='mean')
        data = imputer.fit_transform(data)

        # Normalizacija podataka za bolju konvergenciju
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        if method == 'pca':
            model = PCA(n_components=2 if dim == '2d' else 3)
            reduced_data = model.fit_transform(data)

        elif method == 'svd':
            model = TruncatedSVD(n_components=2 if dim == '2d' else 3)
            reduced_data = model.fit_transform(data)

        elif method == 'mmds':
            max_components = min(20, data.shape[0], data.shape[1])
            if max_components < 2:
                error_message = "ERROR: Insufficient number of samples or dimensions for the MMDS method."
                return render_template('index.html', error_message=error_message)

            pca = PCA(n_components=max_components)
            data_reduced = pca.fit_transform(data)

            input_dim = data_reduced.shape[1]
            hidden_dims = [64, 32]
            output_dim = 2 if dim == '2d' else 3

            model_mmds = MDSNet(input_dim, hidden_dims, output_dim)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            data_tensor = tf.convert_to_tensor(data_reduced, dtype=tf.float32)

            @tf.function
            def train_step(batch_data):
                with tf.GradientTape() as tape:
                    output = model_mmds(batch_data)
                    distance_orig = pairwise_distances(batch_data)
                    distance_proj = pairwise_distances(output)
                    loss = tf.reduce_mean(tf.square(distance_proj - distance_orig))
                gradients = tape.gradient(loss, model_mmds.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model_mmds.trainable_variables))
                return loss

            epochs = 300
            batch_size = 256
            best_loss = float('inf')
            patience, patience_counter = 10, 0

            for epoch in range(epochs):
                permutation = np.random.permutation(data_reduced.shape[0])
                for i in range(0, data_reduced.shape[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_data = tf.gather(data_tensor, indices)
                    loss = train_step(batch_data)
                
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            reduced_data = model_mmds(data_tensor).numpy()

        elif method == 'autoencoder':
            # Smanjenje dimenzionalnosti pomoću PCA
            max_pca_components = min(20, data.shape[1])  
            pca = PCA(n_components=max_pca_components)
            data_reduced = pca.fit_transform(data)
            
            # Definiranje autoencodera
            input_layer = tf.keras.layers.Input(shape=(data_reduced.shape[1],))
            encoded = tf.keras.layers.Dense(10, activation='relu')(input_layer)
            encoded = tf.keras.layers.Dense(5, activation='relu')(encoded)
            encoded = tf.keras.layers.Dense(3, activation='linear')(encoded) 

            decoded = tf.keras.layers.Dense(5, activation='relu')(encoded)
            decoded = tf.keras.layers.Dense(10, activation='relu')(decoded)
            decoded = tf.keras.layers.Dense(data_reduced.shape[1], activation='sigmoid')(decoded)

            autoencoder = tf.keras.models.Model(input_layer, decoded)
            encoder = tf.keras.models.Model(input_layer, encoded)

            # Kompajliranje modela
            autoencoder.compile(optimizer='adam', loss='mae')

            # Treniranje autoencodera
            autoencoder.fit(data_reduced, data_reduced, epochs=200, batch_size=512, shuffle=True, verbose=0)

            # Predviđanje reducirane dimenzije pomoću treniranog autoencodera
            reduced_data = encoder.predict(data_reduced)

        # Generiraj sliku rezultata
        fig = plt.figure()
        if dim == '3d':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        else:
            ax = fig.add_subplot(111)
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1])

        ax.set_title(f' ({method.upper()})')
        plt.tight_layout()

        # Spremi sliku u memoriju
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Spremi sliku na poslužitelj privremeno
        img_path = os.path.join('static', 'result.png')
        with open(img_path, 'wb') as f:
            f.write(img.getvalue())

        # Redirektaj na stranicu s rezultatom
        return redirect(url_for('result'))

    return render_template('index.html', error_message=error_message)

@app.route('/result')
def result():
    return render_template('result.html', image_path='static/result.png')

@app.route('/download')
def download():
    return send_file('static/result.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



