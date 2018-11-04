from flask_wtf import FlaskForm 
from wtforms import StringField, SubmitField, PasswordField, BooleanField, SelectField
from wtforms.validators import DataRequired





class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class TrainForm(FlaskForm):	
	# algorithm = SelectField('Algorithm', choices=[('nb', 'Naive Bayes'), ('dt', 'Decision Tree'), ('rf', 'Random forest'), ('svm', 'SVM'),('logreg','Logistic Regression')])
	algorithm = SelectField('Algorithm', choices=[('nb', 'Naive Bayes'), ('dt', 'Decision Tree'), ('rf', 'Random forest')])

	submit =SubmitField('Train Model')


class PredictForm(FlaskForm):	
	algorithm = SelectField('Algorithm', choices=[('nb', 'Naive Bayes'), ('dt', 'Decision Tree'), ('rf', 'Random forest')])
	clump_thickness = StringField('Clump Thickness', validators=[DataRequired()])
	uniformity_of_cell_size = StringField('Uniformity of Cell Size', validators=[DataRequired()])
	uniformity_of_cell_shape = StringField('Uniformity of Cell Shape', validators=[DataRequired()])
	marginal_adhesion = StringField('Marginal Adhesion', validators=[DataRequired()])
	single_epithelial_cell_size = StringField('Single Epithelial Cell Size', validators=[DataRequired()])
	bare_nuclei = StringField('Bare Nuclei', validators=[DataRequired()])
	bland_chromatin = StringField('Bland Chromatin', validators=[DataRequired()])
	normal_nucleoli = StringField('Normal Nucleoli', validators=[DataRequired()])
	mitoses = StringField('Mitoses', validators=[DataRequired()])
	submit =SubmitField('Predict')