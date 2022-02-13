from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, IntegerField, FloatField
from wtforms.validators import DataRequired


class QueryForm(FlaskForm):
    college = StringField('College Name', validators=[DataRequired()])
    position = StringField('Position', validators=[DataRequired()])
    cluster = StringField('Cluster', validators=[DataRequired()])
    submit = SubmitField('Submit')

class ACTForm(FlaskForm):
    act_score = IntegerField('ACT Score', validators=[DataRequired()])
    position = StringField('Position', validators=[DataRequired()])
    #lat = FloatField('Latitude', validators=[DataRequired()])
    #lon = FloatField('Longitude', validators=[DataRequired()])
    address = StringField('Address', validators=[DataRequired()])
    submit = SubmitField('Submit')