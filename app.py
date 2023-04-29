from flask import  request, render_template,Flask,jsonify
import main
app=Flask(__name__)

@app.route("/home")
def upload():
    return render_template("file_upload.html")

@app.route("/contribution")
def upload1():
    return render_template("contribution.html")



@app.route("/home", methods=["POST"])
def compare():
    text1 = request.form.get('text1')

    result=main.mainfunction(text1)
    return jsonify({ "result":result})

if __name__ == "__main__":
    app.run(debug=True)


    