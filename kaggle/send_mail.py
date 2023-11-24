import os
import ssl
import sys
import smtplib
from email.utils import formataddr
import argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

parse = argparse.ArgumentParser()
parse.add_argument('-s', '--status', choices=['success', 'fail'], type=str)
parse.add_argument('-r', '--recipient', type=str, default='no_one')

def main():
    args = parse.parse_args()

    if args.recipient == 'no_one':
        print('No notification send')
        sys.exit(0)

    sender = 'notification@opc.com'
    sender_name = 'notification@opc.com'
    recepient = args.recipient

    usr_name = ''
    password = ''

    host = ''
    port = None

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