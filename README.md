# mandelpy
Pure python fractal calculation to the max - how fast can python go?

Python is a great language, but being so dynamic has its drawbacks - mainly performance. However tools like numba provide a great way to dramatoically improve performance to C like levels. As an excercise to test various techniques and have a bit of fun, I decided to write a simple mandelbrot app, all in python, and using pygame as the basic framework.

Thw result is very satisfying. I haven't gone as far as using numba's GPU exploits as installing the cuda toolkit is someehat arduous, and also Raspberry Pi is a main target for much of my software, and it cannot 'do' cuda.

So straight CPU bashing it is. With raspberry pi OS 64 bit, most numba facilities are available (including multi-processing). I've also been testing on a Ryzen 7 5800h running Fedora 35. With 16 cpu threads to play on itr really flies (and so so the fans!).

On a Raspberry pi i is limited to 64bit floats, but on my laptop I can run 128bit floats (or maybe they are really only 80 bit floats - but it drills a lot deeper than with 64bit floats. Sadly atm numba doesn't support 128bit floats, so performance takes a big hit though.

## Installation

On A raspberry pi 64 bit OS I add the following packages:
```
pip install numba pyperclip
```
 Oh and it uses pygame 2, which is not in the standard Raspberry pi installation so:
 ```
 pip install 'pygame>=2.0.0'
```

The app needs only the two python files here:
- mandelpy.py is the main app
- mandcalcs.py has various different coings of the mandlbrot calculation

Put these files in a folder of your choice (or git clone this repo).

Put the executable flag on mandelpy.py

## Run the app

From a shell in the right folder:
```
./mandelpy.py
```

Or launch from files.

The app uses the mouse to control zoom and pan and verious keys for other things. just press h to get the help up which explains all the keys.

Note that keys do not auto repeat, and are actioned on key-up

## Some notes
Yes the straight python is slow, buut once numba is used to automatically compile Python to machine code, speed improves dramatically (from 20 secs per frame to 50 frames per second! 0r for Raspberry pi from 1 frame per minute to 10 frames per second)

With 64 bit floats, pixel spacing down to about 1E-16 works before dupolicates start appearing. with 128(?)bit floats, the limit is approx 1E-20.

The speed is way faster tham most madelbrot demo apps, and not far off the speed of specialist fractal apps (unless you are using a machine with a really powerful gpu).

It is surprising that the raw loop version is almost exactly the same speed as the version using a vectorized function - numba appears to be smart enough to 
'improve' the nested python loops to use vector based instructions. In this case at least, the vectorized version is of no benefit.

The only further improvement possible (without going to GPU computation) is probably to use fixed point (integer) maths instead of the raw floating point.

