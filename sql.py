import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Mk_muthu_25",
    database="mysql"
)
if db.is_connected():
    print("Connection established")
    cursor = db.cursor()
    query = "SELECT * FROM patient where name = 'muthu'"
    cursor.execute(query)
    result = cursor.fetchall()
    for row in result:
        print(row)
    cursor.close()
    db.close()

