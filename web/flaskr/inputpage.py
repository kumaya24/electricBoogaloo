import os
from flask import (
    Blueprint, flash, redirect, render_template, request, url_for
)
from flaskr.db import get_db
from flaskr.program import large_program

bp = Blueprint('inputpage', __name__, url_prefix='/')

@bp.route('/', methods=['GET', 'POST'])
def home():
    result = None
    db = get_db()

    if request.method == 'POST':
        user_input = request.form.get('body', '')  # Safely get form data

        if not user_input:
            flash(f"An error occurred: {str(e)}")
        else:
            try:
                # Store entry into db
                db.execute('INSERT INTO post (body) VALUES (?)', (user_input,))
                db.commit()
                flash('Message saved successfully!')

                # Try to run the model
                result = large_program(user_input)
                flash('Results is created.')
            except Exception as e:
                db.rollback()  # Ensure consistency in case of error
                flash(f"An error occurred: {str(e)}")

    # Retrieve all posts from the database
    posts = db.execute('SELECT body FROM post').fetchall()

    # Render the template with the retrieved posts
    return render_template('base.html', posts=posts, result=result)
