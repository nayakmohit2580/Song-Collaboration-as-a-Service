from flask import  redirect, render_template, flash
from songselection import app, db, spotifyclient, recommendations
from songselection.forms import CreateRoom, JoinRoom,RoomForm


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/createroom', methods=['GET','POST'])
def createroom():
    form = CreateRoom()
    form.genre.choices = spotifyclient.SpotifyClient().get_genre_seeds()
    if form.validate_on_submit():
        roomid=form.roomid.data
        roomname=form.roomid.data
        password=form.password.data
        genre = form.genre.data

        cur = db.connection.cursor()
        flag = True
        query = 'SELECT * FROM Rooms WHERE RoomID="' + roomid + '"'
        cur.execute(query)
        db.connection.commit()
        for x in cur:
            l = list(x)
            if l[0] == roomid:
                flag = False
        if flag == True:
            cur.execute('INSERT INTO Rooms VALUES (%s,%s,%s,%s)',(roomid,roomname,password,0))
            flash(f'Room Created. Room ID is {form.roomid.data}','success')
            db.connection.commit()
            cur.close()
            add_first_track(roomid, genre)
            return redirect(f'/joinroom')
        else:
            flash('RoomID present. Enter new RoomID', 'danger')
        
        cur.close()

    return render_template('createroom.html', title='Create Room', form=form)

@app.route('/room/<room_id>/<user>/<track_number>', methods=['GET','POST'])
def room(room_id, user, track_number):
    cur = db.connection.cursor()
    query = 'SELECT Roomname FROM Rooms WHERE RoomID="{roomid}"'.format(roomid=room_id)
    cur.execute(query)
    track_number = int(track_number)
    track_id = fetch_track_id(room_id, track_number)
    user_number = get_user_number(room_id)
    print(f'Room: {room_id}, User: {user}, Track#: {track_number}, TrackId: {track_id}, User#: {user_number}')
    
    for x in cur:
        l = list(x)
        break
    roomname = l[0]

    form = RoomForm()
    value = form.field.data
    
    if form.validate_on_submit():
        if value == "like":
            flag = False
            query = 'Select LikeCount FROM Reactions WHERE RoomID="'+room_id+'" AND TrackID="'+track_id +'"'
            cur.execute(query)
            for x in cur:
                l = list(x)
                flag = True
                break
            if flag == False:
                query = 'INSERT INTO Reactions VALUES (%s,%s,%s,%s,%s,%s)'
                cur.execute(query,(room_id,track_id,track_number,1,0,0))
            elif flag == True:
                likes = l[0] + 1
                query = 'UPDATE Reactions SET LikeCount={like} WHERE RoomID="{roomid}" AND TrackID="{track}"'.format(like=likes,roomid=room_id,track=track_id)
                cur.execute(query)
        if value == "dislike":
            flag = False
            query = 'Select DislikeCount FROM Reactions WHERE RoomID="'+room_id+'" AND TrackID="'+track_id+'"'
            cur.execute(query)
            for x in cur:
                l = list(x)
                flag = True
                break
            if flag == False:
                query = 'INSERT INTO Reactions VALUES (%s,%s,%s,%s,%s,%s)'
                cur.execute(query,(room_id,track_id,track_number,0,1,0))
            elif flag == True:
                dislikes = l[0] + 1
                query = 'UPDATE Reactions SET DislikeCount={dislike} WHERE RoomID="{roomid}" AND TrackID="{track}"'.format(dislike=dislikes,roomid=room_id,track=track_id)
                cur.execute(query)
        if value == "neutral":
            flag = False
            query  = 'Select NeutralCount FROM Reactions WHERE RoomID="'+room_id+'" AND TrackID="'+track_id+'"'
            cur.execute(query)
            for x in cur:
                l = list(x)
                flag = True
                break
            if flag == False:
                query = 'INSERT INTO Reactions VALUES (%s,%s,%s,%s,%s,%s)'
                cur.execute(query,(room_id,track_id,track_number,0,0,1))
            elif flag == True:
                neutral = l[0] + 1
                query = 'UPDATE Reactions SET NeutralCount={neutral} WHERE RoomID="{roomid}" AND TrackID="{track}"'.format(neutral=neutral,roomid=room_id,track=track_id)
                cur.execute(query)
        if form.logout.data:
            query = 'UPDATE Users SET Status=False where Username="{user}" AND RoomID="{roomid}"'.format(user=user,roomid=room_id)
            cur.execute(query)
            db.connection.commit()
            return render_template('home.html')
        if form.nextsong.data:
            if track_number == fetch_track_number(room_id):
                add_recommendation(room_id)
                incrementtrack(room_id)
            return redirect(f'/room/{room_id}/{user}/{track_number + 1}')
    db.connection.commit()      
    return render_template('room.html', title=f'Room {room_id}', track_id=track_id, room_name=roomname,form=form, track_number=track_number, user_number=user_number)

def fetch_track_number(room_id):
    cur = db.connection.cursor()
    query = 'SELECT TrackNumber FROM Rooms WHERE RoomID="{roomid}"'.format(roomid=room_id)
    cur.execute(query)
    row = cur.fetchall()
    cur.close()
    db.connection.commit()
    return row[0][0]

def fetch_track_id(room_id, track_number):
    cur = db.connection.cursor()
    query = f'SELECT TrackID FROM Reactions WHERE RoomID="{room_id}" AND TrackNumber="{track_number}"'
    cur.execute(query)
    row = cur.fetchall()
    cur.close()
    db.connection.commit()
    return row[0][0]

def incrementtrack(room_id):
    cur = db.connection.cursor()
    query = 'SELECT TrackNumber FROM Rooms WHERE RoomID="{roomid}"'.format(roomid=room_id)
    cur.execute(query)
    row = cur.fetchall()
    num = row[0][0] + 1
    query = 'UPDATE Rooms SET TrackNumber={num} WHERE RoomID="{roomid}"'.format(num=num,roomid=room_id)
    cur.execute(query)
    cur.close()
    db.connection.commit()

def add_first_track(room_id, genre):
    cur = db.connection.cursor()
    query = 'INSERT INTO Reactions VALUES (%s,%s,%s,%s,%s,%s)'
    track_id = recommendations.get_recommendation_from_genre(genre)
    cur.execute(query, (room_id, track_id, 0, 0, 0, 0))
    cur.close()
    db.connection.commit()

def add_recommendation(room_id):
    track_number = fetch_track_number(room_id)
    track_id = recommendations.get_recommendation(room_id, track_number)
    query = 'INSERT INTO Reactions VALUES (%s,%s,%s,%s,%s,%s)'
    cur = db.connection.cursor()
    cur.execute(query, (room_id, track_id, track_number + 1, 0, 0, 0))
    cur.close()
    db.connection.commit()

def get_user_number(room_id):
    cur = db.connection.cursor()
    query = 'SELECT COUNT(Username) FROM Users WHERE RoomID="{roomid}"'.format(roomid=room_id)
    cur.execute(query)
    row = cur.fetchall()
    cur.close()
    db.connection.commit()
    return row[0][0]


@app.route('/joinroom', methods=['GET','POST'])
def joinroom():
    form = JoinRoom()
    if form.validate_on_submit():
        roomid=form.roomid.data
        password=form.password.data
        username = form.username.data
        cur = db.connection.cursor()
        flag = False
        query = 'SELECT * FROM Rooms WHERE RoomID="' + roomid + '"' + 'AND Password="' + password + '"'
        cur.execute(query)
        db.connection.commit()
        for x in cur:
            l = list(x)
            if l[0] == roomid and l[2] == password:
                flag = True
        query = 'SELECT * FROM Users WHERE Username="' + username + '"' +  'AND Status=1'
        cur.execute(query)
        db.connection.commit()
        for x in cur:
            l = list(x)
            if l[0]==username and l[2]==1 and l[1]!=roomid:
                flash('User Active in another room', 'danger')
                return render_template('joinroom.html', title='Join Room', form=form)


        if flag == True:
            userpresent = False
            query = 'SELECT * FROM Users WHERE Username="' + username + '"' +  'AND RoomID="' + roomid + '"'
            cur.execute(query)
            db.connection.commit()
            for x in cur:
                userpresent = True
            if userpresent == True:
                query = 'UPDATE Users SET Status=True WHERE Username="' + username + '" AND RoomID="' + roomid +'"'
                cur.execute(query)
                db.connection.commit()
            else:
                cur.execute('INSERT INTO Users VALUES (%s,%s,True)',(username,roomid))
                db.connection.commit()
            #flash('Login Successful','success')
            cur.close()
            return redirect(f'/room/{roomid}/{username}/{fetch_track_number(roomid)}')
        else:
            flash('Login Unsuccessful. Please check RoomID and Password', 'danger')
    return render_template('joinroom.html', title='Join Room', form=form)