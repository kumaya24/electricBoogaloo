import os
from flask import (
    Blueprint, flash, redirect, render_template, request, url_for
)
from flaskr.db import get_db

bp = Blueprint('inputpage', __name__)

@bp.route('/', methods=['GET', 'POST'])
def home():
    db = get_db()

    if request.method == 'POST':
        body = request.form.get('body', '')  # Safely get form data
        error = None

        if not body:
            error = 'Entry is required.'
        else:
            try:
                db.execute('INSERT INTO post (body) VALUES (?)', (body,))
                db.commit()
                flash('Message saved successfully!')
            except Exception as e:
                db.rollback()  # Ensure consistency in case of error
                flash(f"An error occurred: {str(e)}")

    # Retrieve all posts from the database
    posts = db.execute('SELECT body FROM post').fetchall()

    # Render the template with the retrieved posts
    return render_template('base.html', posts=posts)


