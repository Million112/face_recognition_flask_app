from flask import Flask
from app import views 

app = Flask(__name__)

app.add_url_rule(
    rule='/', 
    endpoint='home', 
    view_func=views.index,
    methods=['GET', 'POST']  # 🔥 Cho phép POST để upload ảnh
)

if __name__ == '__main__':
    app.run(debug=True)
