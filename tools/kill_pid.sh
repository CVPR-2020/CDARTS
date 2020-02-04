ps -ef | grep python | grep augment.py | awk {'print $2'} | xargs kill
ps -ef | grep python | grep search.py | awk {'print $2'} | xargs kill
ps -ef | grep python | grep pretrain.py | awk {'print $2'} | xargs kill