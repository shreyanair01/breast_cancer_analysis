from flask import Flask, request, jsonify, render_template, url_for, flash, request, redirect
from forms import LoginForm, TrainForm, PredictForm

import pandas as pd
import numpy as np
# from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, classification_report
# from sklearn import datasets
from sklearn.externals import joblib
import pickle
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)



app.config['SECRET_KEY'] = '023e64a6964e4d8e241453976b5fc05a'

@app.route('/', methods= ['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash(f'Logged in!','success')
        return redirect(url_for('home'))
    return render_template('login.html',title = 'Predict', form=form)





@app.route('/home')
def home():
    # return 'Hello World’
    return render_template('home.html')


@app.route('/about')
def about():
    # return 'Hello World’
    return render_template('about.html',title = 'About')


@app.route('/explore-dataset')
def explore():
    data = pd.read_csv('./static/breastcancer.csv')
    # data.plot(kind='scatter',x=data['Class'],y=data['Clump Thickness'], color = 'r') 
    # plt.savefig('./static/images/new_data_plot.png')
    # corr = (data.corr(method = 'pearson')).to_json()
    corr = (data.corr(method = 'pearson'))
    corr.to_csv("./static/Correlation.csv")
    # print(corr)
    jsondata = data.to_json()
    return render_template('explore_dataset.html',title = 'Explore Data', df = jsondata, url="./static/images/new_data_plot.png", corr = corr)




@app.route('/train', methods=['POST','GET'])
def train():

    # persist model
    # joblib.dump(clf, 'model.pkl')


    cm = [[0,0],[0,0]]
    score = 0
    precision = 0
    recall = 0
    f1score = 0
    form = TrainForm()
    if request.method == 'POST' :
        # print(request.form['algorithm'])


        # read breast cancer data set
        breast_cancer_df = pd.read_csv('./static/breastcancer.csv')
        # if breast_cancer_df[""]
        
        # print(breast_cancer_df[breast_cancer_df["Class"]=='Benign'])

        mapping = {'Malignant': 1, 'Benign': 0}

        breast_cancer_df.replace({'Class': mapping}, inplace = True)

        X = breast_cancer_df.iloc[:,:-1].values
        y = breast_cancer_df.iloc[:,-1].values



        mask = X[:,5]== '?'
        X[mask] = 'NaN'

        

        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(X[:,0:9])
        X[:,0:9] = imputer.transform(X[:,0:9])
        


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        if request.form['algorithm'] == 'nb':
            # # fit model
            clf =  GaussianNB()
        elif request.form['algorithm'] == 'dt':
            # # fit model
            clf =  DecisionTreeClassifier()
        elif request.form['algorithm'] == 'rf':
            # # fit model
            clf =  RandomForestClassifier()
        # elif request.form['algorithm'] == 'svm':
        #     # # fit model
        #     clf =  SVC()
        # elif request.form['algorithm'] == 'logreg':
        #     # # fit model
        #     clf =  LogisticRegression()
            
        clf.fit(X_train, y_train)

         # persist model
        joblib.dump(clf, 'model.pkl')



        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)

        # print(classification_report(y_test,y_pred))
        score = clf.score(X_test,y_test)
        precision = precision_score(y_test, y_pred)
        recall= recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)

       
        test = [[5,1,1,1,2,1,3,1,1]]
        # ,[8,10,10,8,7,10,9,7,1]
        prediction = clf.predict_proba(test)
        print(prediction)
        # flash(f'Model Created '+ str(request.form['algorithm']) + str(clf.score(X_test,y_test))+ str(cm),'success')
        flash(f'Model Created '+ str(request.form['algorithm']) ,'success')
        # return redirect(url_for('train',cm = cm))
        return render_template('train_model.html',title = 'Train Model', form = form, cm = cm, accuracy = score, precision= precision, recall=recall,f1score=f1score)



    return render_template('train_model.html',title = 'Train Model', form = form,cm =cm, accuracy = score, precision= precision, recall=recall,f1score=f1score)






@app.route('/train_r', methods=['GET'])
def trained_results():
    render_template('model_results.html')








@app.route('/predict', methods=['POST','GET'])
def predict():
    # get iris object from request
    # X = request.get_json()
    # X = [[float(X['sepalLength']), float(X['sepalWidth']), float(X['petalLength']), float(X['petalWidth'])]]

    # # read model
    # clf = joblib.load('model.pkl')
    # probabilities = clf.predict_proba(X)
    probabilities = 0


    form = PredictForm()
    if request.method == 'POST' :
        flash(f'Prediction Generated','success')

        clf = joblib.load('model.pkl')

    



        # algorithm = request.form['algorithm']
        clump_thickness = request.form['clump_thickness']
        uniformity_of_cell_size = request.form['uniformity_of_cell_size']
        uniformity_of_cell_shape = request.form['uniformity_of_cell_shape']
        marginal_adhesion = request.form['marginal_adhesion']
        single_epithelial_cell_size = request.form['single_epithelial_cell_size']
        bare_nuclei = request.form['bare_nuclei']
        bland_chromatin = request.form['bland_chromatin']
        normal_nucleoli = request.form['normal_nucleoli']
        mitoses = request.form['mitoses']

        
        # X = np.array([[clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, marginal_adhesion, single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]])
        # prediction = clf.predict_proba(X)
        # pred_df = pd.DataFrame(prediction)
        # pred_df.to_csv("./static/piedata.csv")

        test = np.array([[5,1,1,1,2,1,3,1,1]])
        # test = np.array([[8,10,10,8,7,10,9,7,1]])
        prediction = clf.predict_proba(test)


        
        print(prediction)
        return render_template('predict.html',title = 'Predict', form=form, probabilities = prediction)

    return render_template('predict.html',title = 'Predict', form=form, probabilities=[])

# def go_to():
#     render_template('visualpredict.html',title = 'Predict')



@app.route('/graphs')
def graph_features():
    return render_template('graphs.html')
    # webbrowser.open_new_tab(ur






if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
