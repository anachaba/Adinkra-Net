#from tensorflow import keras tensorflow==2.12.0 numpy==1.23.4 Flask==2.2.2
#from flask import Flask, render_template, request
import numpy as np
import os

class_names1 = ['Aban','Abe_Dua','Adikrahene_Dua','Adinkrahene','Adwera','Adwo','Agyinduwura','Akoben','Akofena','Akokonan','Akoma','Akoma_Ntoso','Ananse_Ntontan','Ani_Bere','Asase_Ye_Duru','Aya','Bese_Saka','Biribi_Wo_Soro','Bi_Nnka_Bi','Boa_Me_Na_Me_Boa_Wo','Dame_Dame','Denkyem','Dono','Duafe','Dwannimmen','Eban','Epa','Ese_Ne_Tekrema','Fafanto','Fawohudie','Fihankra','Fofo','Funtumfunafu_Denkyem_Funafu','Gye_Nyame','Hwemudua','Hye_Won_Hye','Kae_Me','Kete_Pa','Kintinkantan','Kojo_Baiden','Kontire_Ne_Akwamu','Krado','Kramo_Bone','Kuntinkantan','Kwatakye_Atiko','Mako','Mate_Masie','Mframadan','Mmere_Dane','Mmomudwan','Mmusuyidee','Mpatapo','Mpuannum','Nea_Onnim_No_Sua_A_Ohu','Nea_Ope_Se_Nkrofoo_Ye_Ma_Wo_No-_Ye_Saa_Ara_Ma_won','Nea_Ope_Se_Obedi_Hene','Nkonsonkonson','Nkontim','Nkuma_Kese','Nkyimu','Nkyinkyim','Nnonnowa','Nsaa','Nserewa','Nsoromma','Nyame_Akruma','Nyame_Biribi_Wo_Soro','Nyame_Dua','Nyame_Nnwu_Na_Mawu','Nyame_Nti','Nyame_Ye_Ohene','Nyansapo','Nya_Abotere','Odo_Nyera_Fie_Kwan','Ohene','Ohene_Aniwa','Ohene_Tuo','Ohen_Adwae','Okodee_Mmowere','Okuafo_Pa','Onyakopon_Adom_Nti_Biribiara_Beye_Yie_African_Adinkra_Weddin','Onyakopon_Aniwa','Onyakopon_Ne_Yen_Ntena','Osidan','Osram','Osram_Ne_Nsoromma','Owo_Foro_Adobe','Owuo_Atwedee','Owuo_Kum_Nyame','Pa_Gya','Sankofa','Sepow','Sesa_Woruban','Sunsum','Tabon','Tamfo_Bebre','Tumi_Te_Se_Kosua','Tuo_Ne_Akofena','Wawa_Aba','Wo_Nsa_Da_Mu_A','Wuforo_Dua_Pa_A']


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
    app.run(debug=True)