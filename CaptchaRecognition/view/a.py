# coding: utf-8
from flask import Flask, render_template, redirect, url_for, session
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from wtforms import SubmitField
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
import os
from multiprocessing import Pool
from cnn import cnn_model


basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
bootstrap = Bootstrap(app)


class PhotoForm(FlaskForm):
    photo = FileField('', validators=[FileRequired(), FileAllowed(['jpg', 'png'], 'Images only!')])
    submit = SubmitField('解析')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = PhotoForm()
    if form.validate_on_submit():
        f = form.photo.data
        filename = secure_filename(f.filename)
        img_path = basedir+'/'+filename
        f.save(img_path)
        p = Pool()
        r = p.apply_async(cnn_model, (img_path,))
        
        session['result'] = r.get()
        # session['result'] = cnn_model(img_path)
        os.remove(img_path)
        return redirect(url_for('index'))
    return render_template('index.html', form=form, result=session.get('result'))


if __name__ == '__main__':
    app.run(debug=True)