retz-client load-app -A nushio64 \
  --container docker --image nushio3/chainer-semi \
  -F https://github.com/pfnet/chainer/archive/v1.19.0.tar.gz \
  -F https://raw.githubusercontent.com/space-weather-KU/chainer-semi/master/learn-sun/nushio3/01-get-sun-image.py \
  -F https://raw.githubusercontent.com/space-weather-KU/chainer-semi/master/learn-sun/nushio3/04-perfect-memoization-base64.py \
  --user root
