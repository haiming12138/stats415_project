#!/bin/sh

unset recipient mode
while getopts 'r:m:' c
do
  case $c in
    r) recipient=$OPTARG ;;
    m) mode=$OPTARG ;;
  esac
done

[ -z "$recipient" ] && recipient="haiming@umich.edu"

print_result()
{   
    if [ $? = 0 ]
    then
        echo 'Execution Success'
    else
        echo 'Execution Fail'
    fi
}

notify_result()
{
    if [ $? = 0 ]
    then
        python3 send_mail.py -s success -r $recipient
    else
        python3 send_mail.py -s fail -r $recipient
    fi
}

if [ "$mode" = "setup" ]
then
    pip install -r requirements.txt
    print_result
elif [ "$mode" = "xgb" ]
then
	python3 train_xgb.py
    python3 make_submission.py -m xgb
    notify_result
elif [ "$mode" = "linear" ]
then
	python3 train_linear.py
    python3 make_submission.py -m linear
    notify_result
elif [ "$mode" = "optim" ]
then
	python3 iterm_optim.py
    notify_result
else
    echo "Invalid Arguments"
fi