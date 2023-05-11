from tensorflow import keras
from flask import Flask, render_template, request,make_response
import numpy as np
import os


class_names1 = ['Aban','Abe Dua','Adikrahene Dua','Adinkrahene','Adwera','Adwo','Agyinduwura','Akoben','Akofena','Akokonan','Akoma','Akoma Ntoso','Ananse Ntontan','Ani Bere','Asase Ye Duru','Aya','Bese Saka','Biribi Wo Soro','Bi Nnka Bi','Boa Me Na Me Boa Wo','Dame Dame','Denkyem','Dono','Duafe','Dwannimmen','Eban','Epa','Ese Ne Tekrema','Fafanto','Fawohudie','Fihankra','Fofo','Funtumfunafu Denkyem Funafu','Gye Nyame','Hwemudua','Hye Won Hye','Kae Me','Kete Pa','Kintinkantan','Kojo Baiden','Kontire Ne Akwamu','Krado','Kramo Bone','Kuntinkantan','Kwatakye Atiko','Mako','Mate Masie','Mframadan','Mmere Dane','Mmomudwan','Mmusuyidee','Mpatapo','Mpuannum','Nea Onnim No Sua A Ohu','Nea Ope Se Nkrofoo Ye Ma Wo No Ye Saa Ara Ma won','Nea Ope Se Obedi Hene','Nkonsonkonson','Nkontim','Nkuma Kese','Nkyimu','Nkyinkyim','Nnonnowa','Nsaa','Nserewa','Nsoromma','Nyame Akruma','Nyame Biribi Wo Soro','Nyame Dua','Nyame Nnwu Na Mawu','Nyame Nti','Nyame Ye Ohene','Nyansapo','Nya Abotere','Odo Nyera Fie Kwan','Ohene','Ohene Aniwa','Ohene Tuo','Ohen Adwae','Okodee Mmowere','Okuafo Pa','Onyakopon Adom Nti Biribiara Beye Yie African Adinkra Weddin','Onyakopon Aniwa','Onyakopon Ne Yen Ntena','Osidan','Osram','Osram Ne Nsoromma','Owo Foro Adobe','Owuo Atwedee','Owuo Kum Nyame','Pa Gya','Sankofa','Sepow','Sesa Woruban','Sunsum','Tabon','Tamfo Bebre','Tumi Te Se Kosua','Tuo Ne Akofena','Wawa Aba','Wo Nsa Da Mu A','Wuforo Dua Pa A']


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = keras.models.load_model('./models/AdinkraNet_model.h5')

class_names = class_names1

app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image.save('te.jpg')

    img = keras.preprocessing.image.load_img('te.jpg', target_size=IMG_SIZE)

 
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    prediction = model.predict(img_array)

    index = np.argmax(prediction)
    class_name = class_names[index]
    os.remove('te.jpg')

    # Render the predict.html template with the class name
    return render_template('predict.html', class_name=class_name)

# Run the app
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')