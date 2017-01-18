retz-client load-app -A movie-predict-nushio \
  --container docker --image nushio3/chainer-semi \
  -F https://raw.githubusercontent.com/space-weather-KU/chainer-semi/master/learn-sun/nushio3/13-simple-moviepredict.py \
  --user root
