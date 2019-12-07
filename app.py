import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle
import re
import string
import pandas as pd
import numpy as np

app= Flask(__name__)
model= pickle.load(open('model.pkl','rb'))
cv= pickle.load(open('vector.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    message=request.form['message']
    text=message
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[''""...]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('[^a-z, ,]','',text)
    ddd=pd.DataFrame(index=range(0,1))
    ddd['Message']=text
    mess_bow=cv.transform(ddd.Message[0:1])
    message_co= mess_bow.toarray()
    input_variables= message_co
    final_features = input_variables
    prediction = model.predict(final_features)
    return render_template('main.html',result=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
