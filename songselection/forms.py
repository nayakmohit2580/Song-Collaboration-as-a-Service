from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, HiddenField, SelectField, RadioField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
import secrets

def randomstring():
        return secrets.token_hex(10)

class CreateRoom(FlaskForm):
    password = PasswordField('Password',validators=[DataRequired(),Length(min=8,max=14)])
    confirm_password = PasswordField('Confirm Password',validators=[DataRequired(),Length(min=8,max=14),EqualTo('password')])
    submit = SubmitField('Submit')
    roomid = StringField('Room ID', validators=[DataRequired(),Length(min=2,max=20)])
    genre = SelectField('Genre', validators=[DataRequired()])


    '''def validate_roomid(self,roomid):
        roomids = Room.query.filter_by(roomid=roomid.data).first()
        if roomids:
            raise ValidationError('Room ID exists')'''


class JoinRoom(FlaskForm):
    roomid = StringField('Room ID',validators=[DataRequired(),Length(min=2,max=20)])
    username = StringField('Username',validators=[DataRequired(),Length(min=2,max=20)])
    password = PasswordField('Password',validators=[DataRequired(),Length(min=8,max=14)])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Submit')


class RoomForm(FlaskForm):
    field = HiddenField('Field')
    logout = SubmitField('Logout')
    nextsong = SubmitField('Next', id='next')




