from flask import Flask, render_template, request
from system import genre_set, movies_dict, system1, system2, rated_movies
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html', movies=movies_dict)

@app.route("/system1", methods=["POST", "GET"])
def sys1():
    if request.method == 'POST':
        selected_genre = request.form['submit_button']
        selected_m_dict = system1(selected_genre)
        
        return render_template('system1.html', genres=genre_set, movies=selected_m_dict)
    return render_template('system1.html', genres=genre_set)



@app.route("/system2", methods=["POST", "GET"])
def sys2():
    if request.method == 'POST':

        res = []
        for k in rated_movies.keys():
            select = request.form.get(k)
            if select=='0':
                res.append(np.nan)
            else:
                res.append(int(select))
        selected_m_dict = system2(res)
        
        return render_template('system2.html',  select_movies=selected_m_dict)
    return render_template('system2.html', movies=rated_movies)

