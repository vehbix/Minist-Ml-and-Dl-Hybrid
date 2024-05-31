Minist-ML-and-DL-Hybrid
Overview
This project combines Machine Learning (ML) and Deep Learning (DL) techniques to classify the MNIST dataset, which consists of handwritten digits. The hybrid approach aims to leverage the strengths of both ML and DL to achieve higher accuracy in digit classification.

Table of Contents
Introduction
Installation
Usage
Project Structure
Detailed Project Content
Results
Contributing
License
Introduction
The MNIST dataset is a popular benchmark in the field of image processing and machine learning. This project explores a hybrid approach, combining traditional machine learning algorithms with modern deep learning techniques, to improve the classification performance on the MNIST dataset.

Installation
Clone the repository:
git clone https://github.com/vehbix/Minist-Ml-and-Dl-Hybrid.git
Navigate to the project directory:
cd Minist-Ml-and-Dl-Hybrid
Install the required dependencies:
pip install -r requirements.txt
Usage
To run the hybrid model:

Open the Jupyter Notebook ministHybrid.ipynb.
Execute the cells sequentially to train and test the models.
Project Structure
ministHybrid.ipynb: Jupyter Notebook containing the implementation of the hybrid model.
Detailed Project Content
Data Loading and Preprocessing
The notebook begins with loading the MNIST dataset using popular libraries such as TensorFlow/Keras. Data preprocessing steps typically include normalization and reshaping of images.

# Example code for loading data
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
Model Building
Machine Learning Model
A traditional ML model, specifically a Random Forest, is trained on the dataset.

# Example code for a Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train.reshape(-1, 784), y_train)
Deep Learning Model
A Convolutional Neural Network (CNN) is constructed using a deep learning framework.

# Example code for a CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

deep_learning_model = Sequential()
deep_learning_model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
deep_learning_model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(MaxPooling2D(pool_size=2))
deep_learning_model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(MaxPooling2D(pool_size=2))
deep_learning_model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))
deep_learning_model.add(MaxPooling2D(pool_size=2, padding='same'))
deep_learning_model.add(Flatten())
deep_learning_model.add(Dense(256, activation='relu'))
deep_learning_model.add(Dense(10, activation='softmax'))

deep_learning_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
deep_learning_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
Hybrid Model
The predictions from both models are combined, often using techniques like majority voting or a meta-classifier.

# Example code for combining models
ml_predictions = rf_model.predict(x_test.reshape(-1, 784))
dl_predictions = deep_learning_model.predict(x_test)

# Example for majority voting
import numpy as np
final_predictions = np.round((ml_predictions + dl_predictions.argmax(axis=1)) / 2)
Evaluation
The models are evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualizations like confusion matrices and ROC curves are also included.

# Example code for evaluation
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, final_predictions)
print(f'Hybrid Model Accuracy: {accuracy}')
Results
The hybrid model achieved an accuracy of X% on the MNIST test dataset. Detailed performance metrics and visualizations are provided in the notebook.

Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your proposed changes.

License
This project is licensed under the MIT License.


Minist-ML-ve-DL-Hibrit
Genel Bakış
Bu proje, el yazısı rakamları içeren MNIST veri setini sınıflandırmak için Makine Öğrenimi (ML) ve Derin Öğrenme (DL) tekniklerini birleştirir. Hibrit yaklaşım, ML ve DL'in güçlü yönlerinden yararlanarak rakam sınıflandırmada daha yüksek doğruluk sağlamayı amaçlar.

İçindekiler
Giriş
Kurulum
Kullanım
Proje Yapısı
Proje İçeriği Detayları
Sonuçlar
Katkıda Bulunma
Lisans
Giriş
MNIST veri seti, görüntü işleme ve makine öğrenimi alanında popüler bir ölçüt olarak kullanılır. Bu proje, geleneksel makine öğrenme algoritmaları ile modern derin öğrenme tekniklerini birleştirerek MNIST veri seti üzerindeki sınıflandırma performansını artırmayı amaçlamaktadır.

Kurulum
Depoyu klonlayın:
git clone https://github.com/vehbix/Minist-Ml-and-Dl-Hybrid.git
Proje dizinine gidin:
cd Minist-Ml-and-Dl-Hybrid
Gerekli bağımlılıkları yükleyin:
pip install -r requirements.txt
Kullanım
Hibrit modeli çalıştırmak için:
ministHybrid.ipynb Jupyter Notebook dosyasını açın.
Hücreleri sırasıyla çalıştırarak modelleri eğitin ve test edin.
Proje Yapısı
ministHybrid.ipynb: Hibrit modelin uygulanmasını içeren Jupyter Notebook.
Proje İçeriği Detayları
Veri Yükleme ve Ön İşleme
Notebook, MNIST veri setini popüler kütüphaneler kullanarak yükleme ile başlar. Veri ön işleme adımları genellikle normalizasyon ve görüntülerin yeniden şekillendirilmesini içerir.

# Veri yükleme örneği
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
Model Oluşturma
Makine Öğrenimi Modeli
Geleneksel bir ML modeli, özellikle Rastgele Orman, veri seti üzerinde eğitilir.

# Rastgele Orman modeli örneği
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train.reshape(-1, 784), y_train)
Derin Öğrenme Modeli
Derin öğrenme çerçevesi kullanılarak bir Konvolüsyonel Sinir Ağı (CNN) oluşturulur.
# CNN modeli örneği
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

deep_learning_model = Sequential()
deep_learning_model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
deep_learning_model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(MaxPooling2D(pool_size=2))
deep_learning_model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
deep_learning_model.add(MaxPooling2D(pool_size=2))
deep_learning_model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))
deep_learning_model.add(MaxPooling2D(pool_size=2, padding='same'))
deep_learning_model.add(Flatten())
deep_learning_model.add(Dense(256, activation='relu'))
deep_learning_model.add(Dense(10, activation='softmax'))

deep_learning_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
deep_learning_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
Hibrit Model
Her iki modelin tahminleri, çoğunluk oylaması veya bir meta-sınıflandırıcı gibi teknikler kullanılarak birleştirilir.

# Modelleri birleştirme örneği
ml_predictions = rf_model.predict(x_test.reshape(-1, 784))
dl_predictions = deep_learning_model.predict(x_test)

# Çoğunluk oylaması örneği
import numpy as np
final_predictions = np.round((ml_predictions + dl_predictions.argmax(axis=1)) / 2)
Değerlendirme
Modeller, doğruluk, hassasiyet, duyarlılık ve F1 skoru gibi metrikler kullanılarak değerlendirilir. Karışıklık matrisi ve ROC eğrisi gibi görselleştirmeler de dahildir.

# Değerlendirme örneği
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, final_predictions)
print(f'Hibrit Model Doğruluğu: {accuracy}')
Sonuçlar
Hibrit model, MNIST test veri setinde X% doğruluk elde etti. Ayrıntılı performans metrikleri ve görselleştirmeler not defterinde sağlanmıştır.

Katkıda Bulunma
Katkılar memnuniyetle karşılanır! Lütfen bu depoyu fork edin ve önerilen değişikliklerle bir pull request gönderin.

Lisans
Bu proje MIT Lisansı altında lisanslanmıştır.
