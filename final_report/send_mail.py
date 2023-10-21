import os
import ssl
import smtplib
from email.utils import formataddr
import argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

parse = argparse.ArgumentParser()
parse.add_argument('-s', '--status', required=True, 
                   choices=['success', 'fail'], type=str)
parse.add_argument('-r', '--recipient', required=True, type=str)

def main():
    args = parse.parse_args()

    sender = 'notification@opc.com'
    sender_name = 'notification@opc.com'
    recepient = args.recipient

    usr_name = 'ocid1.user.oc1..aaaaaaaaeu3y7cseopcxyxbjhebfepxsbcma26kb6sxtf22c234g55xptd3q@ocid1.tenancy.oc1..aaaaaaaaa4w3s5mjmt6lakqybzxdxa5nlzgmo3agigfda3vs3eabajeddeka.ss.com'
    password = '!Z+2ZEc6$!Gg1gw7A]A;'

    host = 'smtp.email.us-ashburn-1.oci.oraclecloud.com'
    port = 587

    subject = 'execution ' + args.status
    body = ('')

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = formataddr((sender, sender_name))
    msg['To'] = recepient
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(host, port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(usr_name, password)
        server.sendmail(sender, recepient, msg.as_string())
        server.close()
    except Exception as e:
        print(e)
    else:
        print('Notification Send Success')


if __name__ == '__main__':
    main()