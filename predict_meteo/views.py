from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import pickle

# Charger le modèle à partir du fichier .h5
trained_model = load_model('lstm_model.h5')

# Charger le scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Charger le DataFrame
normalized_df = pd.read_csv('data_normalized.csv')

def prediction_view(request):
    last_8_rows = normalized_df.iloc[-8:]  # Prend les 8 dernières lignes
    input_sequence = last_8_rows.to_numpy()
    
    predicted_temperature = predict_and_plot3(input_sequence, trained_model, loaded_scaler)
    
    return render(request, 'prediction.html', {"predicted_temperature": predicted_temperature})


def predict_and_plot3(sequence, model, scaler):
    input_sequence = sequence[:-1]
    input_sequence_reelle = scaler.inverse_transform(sequence)
    predicted_temperature_normalisee = model.predict(np.expand_dims(input_sequence, axis=0))[0][0]
    predicted_temperature_reelle = (predicted_temperature_normalisee * scaler.scale_[-1]) + scaler.mean_[-1]

    hours = ['01H', '04H', '07H', '10H', '13H', '16H', '19H', '22H', 'Prédiction']
    
    plt.figure(figsize=(10, 6))
    plt.plot(hours[:-1], input_sequence_reelle[:, -1], marker='o', linestyle='-', color='blue', label='Températures réelles')
    plt.scatter('Prédiction', predicted_temperature_reelle, color='red', marker='X', label='Température prédite')
    plt.plot(['22H', 'Prédiction'], [input_sequence_reelle[-1, -1], predicted_temperature_reelle], color='red')
    
    for hour, temperature in zip(hours[:-1], input_sequence_reelle[:, -1]):
        plt.text(hour, temperature, f'{temperature:.2f}', fontsize=8, ha='right', va='bottom')

    plt.text('Prédiction', predicted_temperature_reelle, f'{predicted_temperature_reelle:.2f}', fontsize=8, ha='right', va='bottom')
    
    plt.xlabel('Heure')
    plt.ylabel('Température (°C)')
    plt.title('Évolution de la température avec prédiction')
    plt.legend()
    plt.grid()
    
    # Sauvegarder le plot comme image
    plt.savefig("static/images/prediction.png")
    
    return round(predicted_temperature_reelle, 2)



from django.http import JsonResponse
from django.core.mail import EmailMessage
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def send_email(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        email_address = data.get('email')

        if not email_address:
            return JsonResponse({'status': 'error', 'message': 'Adresse e-mail manquante.'}, status=400)

        email = EmailMessage(
            'Votre prédiction de température',
            'Voici votre prédiction de température.',
            'projettutor2023@gmail.com', 
            [email_address],
        )
        
        file_path = "static/images/prediction.png"
        email.attach_file(file_path)
        
        email.send()

        return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error'}, status=405)
