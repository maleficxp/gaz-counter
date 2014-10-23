from flask import Flask, render_template, redirect, url_for, request, flash
from models import sess, Image, dgDigitById, fh
import re
app = Flask(__name__)
app.logger.addHandler(fh)

from datetime import date, datetime, timedelta

def parsePost(post):
    res = dict()
    for k,v in post.items():        
        m = re.match('(.*)\[(.*)\]',k)
        if m :        
            if m.group(1) not in res:
                res[m.group(1)] = dict()
            res[m.group(1)][m.group(2)] = v  
        else:
            res[k] = v    
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # refresh dates 
        return redirect(url_for('index', from_date=request.form['from_date'], to_date=request.form['to_date']))
    else:
        if request.args.get('from_date'):
            from_date=datetime.strptime(request.args.get('from_date'), "%d.%m.%Y")
        else :
            from_date = date.today() + timedelta( weeks = -1 )
        if request.args.get('to_date'):
            to_date=datetime.strptime(request.args.get('to_date'), "%d.%m.%Y")
        else:
            to_date = date.today()
        
        # fetch values 
        images = sess.query(Image).filter(Image.check_time>=from_date, Image.check_time<=to_date+timedelta(days=1), Image.result!='').order_by(Image.check_time).all()
        data = []

        if len(images)>0:
            prev_check = None
            first = images[0]
            last = images[-1]        
            
            # calculate full time interval
            full_interval = last.check_time-first.check_time
            # define min time step
            min_step = full_interval/200 
            # calculate full average
            average_consumption = round((float(last.result)-float(first.result))/100/full_interval.total_seconds() * 60 * 60 * 24,2)
    
    
            for image in images:
                check = {'time':image.check_time, 'value':float(image.result)/100}
                if None != prev_check:
                    # caclulate time delta with previous check 
                    td = check['time']-prev_check['time']
                    # skip very often checks
                    if td<min_step: 
                        continue
                    # caclulate tic average consumption
                    check['diff'] = round(float(check['value']-prev_check['value'])/td.total_seconds() * 60 * 60 * 24, 2)    
                    # set diff value of first check equal to diff value of second check
                    if prev_check['diff']==None:
                        prev_check['diff'] = check['diff']
                else:
                    # first check point
                    check['diff'] = None
                # push chekpoint to series
                data.append(check)
                prev_check = check
        else:
            average_consumption = 0;
            
                    
        unrecognized_images_cnt = sess.query(Image).filter_by(result='').count()
        
        return render_template('index.html', from_date=from_date, to_date=to_date, series=data, average_consumption=average_consumption, unrecognized_images_cnt=unrecognized_images_cnt)

@app.route('/recognize')
def recognize():
    unrecognized_images = sess.query(Image).filter_by(result='').limit(100).all()
    return render_template('recognize.html', images=unrecognized_images)
 
@app.route('/save_digits', methods=['POST'])
def save_digits():
    post = parsePost(request.form)
    images_to_recognize_again = dict()
    
    # save manually recognized digits 
    for digit_id, digit_val in post['result'].items():
        dbdigit = dgDigitById(digit_id)
        if dbdigit.result!=digit_val[0]:
            images_to_recognize_again[dbdigit.image_id] = dbdigit.image        
        dbdigit.result = digit_val[0]
        if 'use_for_training' in post and digit_id in post['use_for_training']:
            dbdigit.use_for_training = True
        else:
            dbdigit.use_for_training = False
    
    # identify saved images again
    for img_id,image in images_to_recognize_again.items():
        image.identifyDigits()
            
    sess.commit()
    return redirect(url_for('recognize'))
    
if __name__ == "__main__":
    app.run(debug=True)
