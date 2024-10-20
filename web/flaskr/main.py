from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from program import large_program  # Assuming this is your outer program

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("Select File", validators=[InputRequired()])
    submit = SubmitField("Upload")

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    result = None  # Initialize the result variable

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)

        # Save the file to the configured folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Call the outer program (large_program) and store the result
            with open(file_path, 'r') as f:
                result = large_program(f)  # Adjust if it expects different input

            flash("File processed successfully!", "success")
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")

    # Render the template with the form and result
    return render_template('index.html', form=form, result=result)

if __name__ == '__main__':
    app.run(debug=True)

