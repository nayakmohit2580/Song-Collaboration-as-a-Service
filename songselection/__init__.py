from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = '0169d14ae726969b8496'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rooms.db' 
app.config['MYSQL_HOST'] = 'colab.cfahydrhpztm.us-east-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'songselection'


#db = SQLAlchemy(app)
db = MySQL(app)
#login_manager = LoginManager(app)


from songselection import routes