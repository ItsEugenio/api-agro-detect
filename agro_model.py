import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Constantes
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_model(model_path):
    """
    Carga un modelo guardado en formato H5
    
    Args:
        model_path: Ruta al archivo .h5 del modelo
    
    Returns:
        model: Modelo cargado
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None

def load_class_names(class_names_path):
    """
    Carga los nombres de las clases desde un archivo de texto
    
    Args:
        class_names_path: Ruta al archivo con los nombres de las clases
    
    Returns:
        class_names: Lista de nombres de clases
    """
    if not os.path.exists(class_names_path):
        print(f"❌ Archivo de nombres de clases no encontrado: {class_names_path}")
        return None
    
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(f"✅ Clases cargadas: {len(class_names)} clases")
        return class_names
    except Exception as e:
        print(f"❌ Error al cargar los nombres de las clases: {e}")
        return None

def load_and_preprocess_image(img_path):
    """
    Carga y preprocesa una imagen para el modelo
    
    Args:
        img_path: Ruta a la imagen
    
    Returns:
        img: Imagen preprocesada
        original_img: Imagen original para visualización
    """
    if not os.path.exists(img_path):
        print(f"❌ Imagen no encontrada: {img_path}")
        return None, None
    
    try:
        # Cargar imagen
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Error al leer la imagen: {img_path}")
            return None, None
        
        # Guardar copia de la imagen original para visualización
        original_img = img.copy()
        
        # Convertir BGR a RGB (OpenCV carga en BGR, TensorFlow espera RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar a las dimensiones esperadas por el modelo
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalizar a [0,1]
        img = img / 255.0
        
        # Expandir dimensiones para el batch (el modelo espera [batch, height, width, channels])
        img = np.expand_dims(img, axis=0)
        
        print(f"✅ Imagen cargada y preprocesada: {img_path}")
        return img, original_img
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")
        return None, None
    
def load_and_preprocess_image_bytes(image_bytes):
    try:
        # Convertir bytes a imagen de OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print(f"❌ Error al decodificar la imagen")
            return None, None

        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        print("✅ Imagen cargada y preprocesada desde bytes")
        return img, original_img
    except Exception as e:
        print(f"❌ Error al procesar la imagen desde bytes: {e}")
        return None, None    

def predict(model, img, class_names):
    """
    Realiza una predicción con el modelo
    
    Args:
        model: Modelo cargado
        img: Imagen preprocesada
        class_names: Lista de nombres de clases
    
    Returns:
        results: Lista de diccionarios con predicciones
    """
    try:
        # Realizar predicción
        predictions = model.predict(img)
        
        # Obtener índices ordenados por confianza (de mayor a menor)
        sorted_indices = np.argsort(predictions[0])[::-1]
        
        # Crear lista de resultados
        results = []
        for idx in sorted_indices:
            results.append({
                "class_name": class_names[idx],
                "confidence": float(predictions[0][idx] * 100)
            })
        
        print(f"✅ Predicción completada")
        return results
    except Exception as e:
        print(f"❌ Error al realizar la predicción: {e}")
        return None

def generate_gradcam(model, img, class_idx):
    """
    Genera un mapa de calor Grad-CAM para visualizar áreas importantes
    
    Args:
        model: Modelo cargado
        img: Imagen preprocesada
        class_idx: Índice de la clase a visualizar
    
    Returns:
        heatmap_img: Imagen con el mapa de calor superpuesto
    """
    try:
        # Encontrar la última capa convolucional
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            print("⚠️ No se encontraron capas convolucionales para Grad-CAM")
            return None
        
        # Crear modelo Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, class_idx]
        
        # Gradientes de la clase con respecto a la salida del feature map
        grads = tape.gradient(loss, conv_outputs)
        
        # Vector de pesos globales mediante pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplicar cada canal por su importancia
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalizar entre 0 y 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Redimensionar heatmap al tamaño de la imagen original
        heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
        
        # Convertir a formato RGB para visualización
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Obtener imagen original
        original_img = (img[0] * 255).astype(np.uint8)
        
        # Superponer el heatmap a la imagen original
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        print(f"✅ Grad-CAM generado")
        return superimposed_img
    except Exception as e:
        print(f"❌ Error al generar Grad-CAM: {e}")
        return None

def visualize_results(original_img, processed_img, results, gradcam=None):
    """
    Visualiza los resultados de la predicción
    
    Args:
        original_img: Imagen original
        processed_img: Imagen procesada
        results: Lista de resultados de predicción
        gradcam: Imagen con Grad-CAM (opcional)
    """
    # Configurar figura
    if gradcam is not None:
        plt.figure(figsize=(15, 6))
    else:
        plt.figure(figsize=(10, 6))
    
    # Mostrar imagen original
    if gradcam is not None:
        plt.subplot(1, 3, 1)
    else:
        plt.subplot(1, 2, 1)
    
    # Convertir de BGR a RGB para visualización
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(original_img)
    plt.title('Imagen Original')
    plt.axis('off')
    
    # Mostrar resultados de predicción
    if gradcam is not None:
        plt.subplot(1, 3, 2)
    else:
        plt.subplot(1, 2, 2)
    
    # Crear texto con resultados
    y_pos = 0.95
    plt.text(0.05, y_pos, "Predicciones:", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    y_pos -= 0.08
    
    for i, result in enumerate(results[:5]):  # Mostrar top 5 resultados
        confidence_str = f"{result['confidence']:.2f}%"
        color = 'green' if i == 0 else 'black'
        weight = 'bold' if i == 0 else 'normal'
        plt.text(0.05, y_pos, f"{i+1}. {result['class_name']}", fontsize=12, 
                color=color, fontweight=weight, transform=plt.gca().transAxes)
        plt.text(0.85, y_pos, confidence_str, fontsize=12, 
                color=color, fontweight=weight, transform=plt.gca().transAxes, 
                horizontalalignment='right')
        y_pos -= 0.06
    
    plt.axis('off')
    plt.title('Resultados de la Predicción')
    
    # Mostrar Grad-CAM si está disponible
    if gradcam is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(gradcam)
        plt.title('Grad-CAM: Áreas Importantes')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    """Función principal"""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Probar modelo H5 con una imagen')
    parser.add_argument('--model', type=str, default='models/agro_detect_model.h5',
                        help='Ruta al archivo .h5 del modelo')
    parser.add_argument('--classes', type=str, default='models/class_names.txt',
                        help='Ruta al archivo con nombres de clases')
    parser.add_argument('--image', type=str, required=True,
                        help='Ruta a la imagen para probar')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generar visualización Grad-CAM')
    
    args = parser.parse_args()
    
    # Cargar modelo
    model = load_model(args.model)
    if model is None:
        return
    
    # Cargar nombres de clases
    class_names = load_class_names(args.classes)
    if class_names is None:
        return
    
    # Cargar y preprocesar imagen
    processed_img, original_img = load_and_preprocess_image(args.image)
    if processed_img is None:
        return
    
    # Realizar predicción
    results = predict(model, processed_img, class_names)
    if results is None:
        return
    
    # Mostrar resultados en consola
    print("\n🔍 Resultados de la predicción:")
    for i, result in enumerate(results[:5]):  # Top 5 resultados
        print(f"{i+1}. {result['class_name']}: {result['confidence']:.2f}%")
    
    # Generar Grad-CAM si se solicita
    gradcam_img = None
    if args.gradcam:
        top_class_idx = class_names.index(results[0]['class_name'])
        gradcam_img = generate_gradcam(model, processed_img, top_class_idx)
    
    # Visualizar resultados
    visualize_results(original_img, processed_img, results, gradcam_img)
    
    print(f"\n✅ Resultado guardado como 'prediction_result.png'")

if __name__ == "__main__":
    main()