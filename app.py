from flask import Flask, request, redirect, url_for
from recommend import load_model, recommend

app = Flask(__name__)
model = load_model()


@app.route("/", methods=["GET", "POST"])
def login():

    user_id = "0"
    k = 10
    likes = ""
    nlikes = ""
    mode = 'random'
    if request.method == "POST":
        user_id = request.form["user_id"]
        mode = request.form["mode"]
        k = int(request.form["k"])
        # mode = 'random'
        print(user_id, mode, k, type(k))
        likes, nlikes, _, _, _ = recommend(model, user_id, k, mode)
        liked_books = "<p>".join(liked_books)
        nliked_books = "<p>".join(nliked_books)

    return f"""
    <!doctype html>
    <title>Get preference-inconsistent book recommendations</title>
    <h1>Get book recommendations</h1>
    <form method=post enctype=multipart/form-data>

    <p> User id              
        <input type=number min=0 step=1 max=45962 name=user_id value={user_id}>
    <p> No of recommendations
        <input type=number min=10 step=1 max=50 name=k value={k}> 
    <div>
    <input type="radio" id="random" name="mode" value="random" checked="checked">
    <label for="random">random</label>
    </div>

    <div>
    <input type="radio" id="interested_only" name="mode" value="top">
    <label for="top">interested_only</label>
    </div>

    <div>
    <input type="radio" id="topic" name="mode" value="top">
    <label for="top">topic</label>
    </div>

    <p><input type=submit value=recommend>
    </form>
    <p> The system recommends {k} books you might be interested in
    <p> =============
    <p> Books you might agree with
    <p>{likes}
    <p> =============
    <p> Books you might disagree with
    <p>{nlikes}

    """


if __name__ == "__main__":
    app.run(debug=True)
