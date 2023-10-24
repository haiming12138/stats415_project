#!/bin/sh

unset recipient mode
while getopts 'r:m:' c
do
  case $c in
    r) recipient=$OPTARG ;;
    m) mode=$OPTARG ;;
  esac
done

[ -z "$recipient" ] && recipient="no_one"

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
    Rscript --vanilla create_data.R
    print_result
elif [ "$mode" = "cleanup" ]
then
    rm -R ./.cache/*
else
	python3 main.py -m $mode
    notify_result
fi