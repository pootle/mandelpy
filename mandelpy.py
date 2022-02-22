#!/usr/bin/env python
import pygame, numpy as np, math
import numba, time, sys
from pygame import surfarray
import pygame.locals as pygconsts
import pygame.mouse as pygmouse
from collections import namedtuple
import pyperclip, json

BLACK = (0, 0, 0)
BLACKSEMI = (0,0,0,128)
WHITE = (255, 255, 255)
RED   = [200,100,80]

class dim1():
    """
    mini helper class with bounds on 1 dimension
    """
    def __init__(self, min_v, max_v=None):
        if isinstance(min_v, type(self)):
            self.min_v = min_v.min_v
            self.max_v = min_v.max_v
            self.size = self.max_v - self.min_v
        else:
            self.setpos(min_v, max_v)

    def pan(self, v, parent = None):
        if parent and v < 0 and self.min_v + v < parent.min_v:
            v = parent.min_v - self.min_v
        if parent and v > 0 and self.max_v + v > parent.max_v:
            v = parent.max_v - self.max_v
        self.min_v += v
        self.max_v += v
        return v != 0

    def zoom(self, scaleby, origin, parent = None):
        if origin < self.min_v:
            left_frac = 0
        elif origin > self.max_v:
            left_frac = 1
        else:
            left_frac = (origin-self.min_v)/self.size
        self.size *= scaleby
        if parent and self.size > parent.size:
            self.size = parent.size
        new_min = origin-self.size*left_frac
        new_max = new_min+self.size
        if parent:
            if new_min < parent.min_v:
                new_max += parent.min_v-new_min
                new_min = parent.min_v
            if new_max > parent.max_v:
                new_min -= new_max - parent.max_v
                new_max = parent.max_v
        self.min_v = new_min
        self.max_v = new_max

    def setpos(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v
        self.size = self.max_v - self.min_v

    @property
    def centre(self):
        return (self.min_v+self.max_v) / 2



class Bounds2d():
    def __init__(self, min_x, max_x=None, min_y=None, max_y=None, parent=None):
        if isinstance(min_x, type(self)):
            self.xs = dim1(min_x.xs)
            self.ys = dim1(min_x.ys)
        else:
            self.xs = dim1(min_x, max_x)
            self.ys = dim1(min_y, max_y)
        self.parent=parent

    @property
    def aspect_ratio(self):
        return self.xs.size/self.ys.size

def makelookup(limit):
    """
    make a lookup table for iteration count to rgb colour to avoid recalculating
    
    Actually 4 lookup tables with offset values out to help make animation smoother
    """
    lut=np.empty((4,limit+1,3), dtype='uint8')
    a = math.pi*2/limit
    for x in range(4):
        zz=x*a/4
        for i in range(limit+1):
            lut[x,i] = ((.5*math.sin(a*i+zz) + .4999)*256, (.5*math.sin(a*i+2.094+zz) + .4999) *256, (.5*math.sin(a*i+4.188+zz) + .4999) *256)
    return lut

help_text = [
    'Mouse controls for this app:',
    '    pan and scroll click left mpouse button and drag the view around. Note it has boundaries, so this',
    "        won't do much until zoomed in",
    '    scroll wheel zooms in and out, centered on the cursor position',
    'Keyboard controls:',
    '    o or p :   switch between straight python and numba jitted python',
    '    i and k:   cycles through the list of different codings for the core calculation',
    '    w and s:   increases or decreases the iteration limit',
    '    x and z:   zoom in / out centered on mouse position',
    '    b      :   switches between jit and plain python version of the code to map from',
    '               iteration counts to rgb colours',
    '    a      :   cycle animation: forwards, backwards or off',
    '    r      :   re calculate every frame or re calculate only on changed view',
    '    j      :   show / hide the stats panel',
    '    h      :   show this help',
    ' CTRL c    :   copy current co-ordinates to clipboard',
    ' CTRL v    :   set location from the clipboard (if it can be parsed...',
    ' This program was originally converted from C++ based on the code described here',
    ' https://www.youtube.com/watch?v=PBvLs88hvJ8'
]

class textpanel():
    """
    class for simple management of several lines of text in a panel than can be changed dynamically
    lines of text are held in a dict and can be updated individually
    """
    def __init__(self, default_font, border = 2, background = (0,0,0,100), r_params = {'fgcolor': (255,255,255)}):
        """
        prepares an empty list of lines and records the default (currently only) font
        """
        self.lines = {}                 # the string surface and rect for each line
        self.def_font = default_font    # the font to use for all text in the panel
        self.border = border            # pixel size of the border
        self.linespace = 3              # extra pixels between lines
        assert len(background) == 4
        self.background = background    # rgba for the panel
        self.default_r_params = r_params
        self.current_surface = None
        
    def add_line(self, name, text, r_params=None):
        assert not name in self.lines
        self.lines[name] = [text, r_params, None, None]
        self._make_text_surf(name)
        self.current_surface = None

    def delete_line(self, name):
        assert name in self.lines
        del(self.lines[name])
        self.current_surface = None

    def update_line(self, name, text, r_params=None):
        assert name in self.lines
        self.lines[name][0] = text
        if not r_params is None:
            self.lines[1] = r_params
        self._make_text_surf(name)
        self.current_surface = None

    def _make_text_surf(self, name):
        t_info = self.lines[name]
        if t_info[1] is None:
            t_info[2], t_info[3] = self.def_font.render(t_info[0], **self.default_r_params)
        else:
            r_par = self.default_r_params.copy()
            r_par.update(t_info[1])
            t_info[2], t_info[3] = self.def_font.render(t_info[0], **self.default_r_params)

    def get_surface(self):
        if self.current_surface is None:
            maxw=max([txtr[3].width for txtr in self.lines.values()])
            heights = [txtr[3].height for txtr in self.lines.values()]
            self.current_surface = pygame.Surface((maxw+self.border*2, sum(heights) + self.border*2 + self.linespace*len(heights)), pygame.SRCALPHA)
            self.current_surface.fill(self.background)
            vpos = self.border
            for tl in self.lines.values():
                lpos = tl[3].move(self.border,vpos-tl[3].top)
                vpos +=lpos.height + self.linespace
                self.current_surface.blit(tl[2], lpos)
        return self.current_surface

class fracman():
    """
    A class to play with fractals and in particular performance of different code for
    core fuuntions
    """
    def __init__(self, universe=Bounds2d(-2.5, 1.00001, -1.25, 1.25), initial_iter=100, enginelist=None):
        """
        universe: sets the bounds for our universe - the window onto the universe cannot exceed these values
                  The 1.00001 value ensures all parts are floats which helps later with numba
        """
        pygame.init()
        self.uni_limits = Bounds2d(universe)                    # save the universe' bounds
        self.cur_univ = Bounds2d(universe,parent = self.uni_limits) # use the universe bounds as initial bounds
        self.set_screen_size(800, int(800/self.cur_univ.aspect_ratio))
        self.display = pygame.display.set_mode((self.winx, self.winy), pygconsts.RESIZABLE)
        pygame.display.set_caption('Mandelbrots in python')       # setup pygame's display
        self.prim_font = pygame.freetype.SysFont(None, 18)
        self.max_iter = initial_iter                            # initial maximum iterations
        self.current_lut = makelookup(min(self.max_iter,128))   # a lookup table to convert function's interger return values to nice colour
        self.enginelist = enginelist                            # list of functions to calculate values
        self.engine_vx = 2                                      # and initial indexes into enginelist
        self.engine_ix = 3                                      # ditto
        self.surf = pygame.surfarray.make_surface(self.npimg)
        self.wheel_zoom_scale = .03
        self.calc_times=[]
        self.blat_times=[]
        self.recalc = True
        self.stats_on = False
        self.force_calc = True
        self.infopanel = textpanel(self.prim_font, background = (50,0,0,100))
        self.infopanel.add_line('max iter', 'max iterations:  %d' % self.max_iter)
        self.infopanel.add_line('fps', 'fps: -' )
        self.infopanel.add_line('engine', 'engine: %s' % self.enginelist[self.engine_vx][self.engine_ix][1])
        self.infopanel.add_line('pxsize', 'pixel size: {:.3E}'.format(self.cur_univ.xs.size / self.winx))
        self.infopanel.add_line('calc', 'calc time: -')
        self.infopanel.add_line('blat', 'blat time: -')

    def set_screen_size(self, win_x, win_y):
        self.winx = win_x
        self.winy = win_y
        self.npimg = np.empty((self.winx,self.winy,3), dtype='uint8')# make a numpy array to put the data for the next image to display
        self.npdata = np.empty((self.winx,self.winy), dtype='uint16')# and an array for raw calculation results

    def run_loop(self):
        running = True
        panactive = False
        jit_blat = True
        c_ctr = 0
        last_anim_time = time.time()
        animate=1
        fpscounter = 0
        fpstimer=time.time()
        fpstext, _ = self.prim_font.render('   fps: ?', WHITE, BLACK)
        logmsgs = 0
        help_on=False
        active_engine = self.enginelist[self.engine_vx][self.engine_ix][0]
        while running:
            wheelzoomcount = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    if panactive:
                        pan_m_currentpos = event.pos
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        mouse_at = event.pos
                        wheelzoomcount -= 1
                    elif event.button == 5:
                        mouse_at = event.pos
                        wheelzoomcount += 1
                    elif event.button == 1:
                        if not panactive:
                            panactive = True
                            pan_m_startpos=event.pos
                            pan_m_currentpos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button==1:
                    panactive = False
                elif event.type == pygame.KEYUP:
                    try:
                        evkey = chr(event.key)
                    except:
                        evkey = None
                        print('what is', event.key)
                    if evkey in ('o','p','i','k'):
                        active_engine = self.change_engine(evkey)
                    elif evkey == 'w' or evkey == 's':
                        if evkey == 'w':
                            self.max_iter = int(self.max_iter*1.5)
                        else:
                            self.max_iter = int(self.max_iter/1.5)
                            if self.max_iter < 4:
                                self.max_iter = 4
                        self.update_lut(self.max_iter)
                        self.infopanel.update_line('max iter', 'max iterations:  %d' % self.max_iter)
                    elif evkey == 'x' or evkey=='z':
                        mx, my = pygmouse.get_pos()
                        if mx < 0:
                            mx = 0
                        if mx >= self.winx:
                            mx = self.win-1
                        if my < 0:
                            my = 0
                        if my >= self.winy:
                            my = self.winy-1
                        mouse_at = mx, my
                        wheelzoomcount += 1 if evkey == 'x' else -1
                    elif evkey == 'r':
                        self.force_calc = not self.force_calc
                    elif evkey == 'b':
                        jit_blat = not jit_blat
                    elif evkey == 'j':
                        self.stats_on = not self.stats_on
                    elif evkey == 'a':
                        if animate == 1:
                            animate = -1
                        else:
                            animate += 1
                    elif evkey == 'h':
                        help_on = not help_on
                        if help_on:
                            help_panel = textpanel(self.prim_font)
                            for ti, tl in enumerate(help_text):
                                help_panel.add_line(ti, tl)
                        else:
                            help_panel = None
                    elif evkey == 'c' and event.mod & pygconsts.KMOD_CTRL:
                        info = {
                            'xloc': self.cur_univ.xs.centre, 'yloc': self.cur_univ.ys.centre,
                            'inc': self.cur_univ.xs.size / self.winx, 'iter_limit': self.max_iter}                            
                        c_string = json.dumps(info, indent=3)
                        pyperclip.copy(c_string)
                    elif evkey == 'v' and event.mod & pygconsts.KMOD_CTRL:
                        c_txt = pyperclip.paste()
                        if not c_txt is None:
                            try:
                                info = json.loads(c_txt)
                                xloc = float(info['xloc'])
                                yloc = float(info['yloc'])
                                pitch = float(info['inc'])
                                iters = int(info['iter_limit'])
                                half = pitch*self.winx/2
                                ax = self.cur_univ.xs
                                ax.setpos(xloc-half, xloc+half)
                                half = pitch*self.winy/2
                                ax = self.cur_univ.ys
                                ax.setpos(yloc-half, yloc+half)
                                self.max_iter = iters
                                self.infopanel.update_line('max iter', 'max iterations:  %d' % self.max_iter)
                                self.infopanel.update_line('pxsize', 'pixel size: {:.3E}'.format(self.cur_univ.xs.size / self.winx))
                                self.recalc=True
                            except:
                                pass
                        else:
                            print('not text')
                elif event.type == pygame.VIDEORESIZE:
                    axis_x = self.cur_univ.xs.centre
                    axis_y = self.cur_univ.ys.centre
                    px_pitch = self.cur_univ.xs.size / self.winx
                    self.set_screen_size(*event.size)
                    x_half = px_pitch * self.winx / 2
                    y_half = px_pitch * self.winy / 2
                    self.cur_univ.xs.setpos(axis_x - x_half, axis_x + x_half)
                    self.cur_univ.ys.setpos(axis_y - y_half, axis_y + y_half)
                    self.recalc=True
                else:
                    pass
#                    print('unknown event %d' % event.type)
            if panactive:
                offs = (pan_m_currentpos[0] - pan_m_startpos[0]) * self.cur_univ.xs.size / self.winx
                if self.cur_univ.xs.pan(-offs, self.cur_univ.parent.xs):
                    self.recalc=True
                offs = (pan_m_currentpos[1] - pan_m_startpos[1]) * self.cur_univ.ys.size / self.winy
                if self.cur_univ.ys.pan(-offs, self.cur_univ.parent.ys):
                    self.recalc = True
                pan_m_startpos = pan_m_currentpos
            if wheelzoomcount != 0:
                scaleby = 1+self.wheel_zoom_scale * wheelzoomcount
                axis_x, axis_y = self.mouse_to_universe(mouse_at)
                self.cur_univ.xs.zoom(scaleby, axis_x, self.uni_limits.xs)
                px_pitch = self.cur_univ.xs.size / self.winx
                y_base = axis_y - mouse_at[1] * px_pitch
                self.cur_univ.ys.setpos(y_base, y_base+px_pitch*self.winy)
                self.infopanel.update_line('pxsize','pixel size: {:.3E}'.format(self.cur_univ.xs.size / self.winx))
                self.recalc=True
            if running:
                win_w, win_h = self.display.get_size()
                if win_w == self.winx and win_h == self.winy:
                    if (self.recalc or self.force_calc) and active_engine:
                        st = time.time()
                        active_engine(self.npdata, self.cur_univ.xs.min_v, self.cur_univ.xs.max_v, self.cur_univ.ys.min_v, self.cur_univ.ys.max_v, self.winx, self.winy, self.max_iter)
                        self.recalc = False
                        self.calc_times.append(time.time()-st)
                        if len(self.calc_times) > 10:
                            self.calc_times.pop(0)
                    st = time.time()
                    if jit_blat:
                        j_blat(self.npdata, self.npimg,self.current_lut, c_ctr)           
                    else:
                        blat(self.npdata, self.npimg,self.current_lut, c_ctr)
                    self.blat_times.append(time.time()-st)
                    if len(self.blat_times) > 10:
                        self.blat_times.pop(0)
                    surfarray.blit_array(self.display, self.npimg)
                    nowtime = time.time()
                    fpscounter += 1
                    if fpstimer+1 < nowtime:
                        if self.stats_on:
                            self.infopanel.update_line('fps', '   fps: %5.2f' % (fpscounter / (nowtime-fpstimer)))
                        fpstimer = nowtime
                        fpscounter = 0
                    if help_on:
                         self.display.blit(help_panel.get_surface(), (2,2))
                    elif self.stats_on:
                        self.infopanel.update_line('calc', 'calc time: %5.3f' % np.mean(self.calc_times))
                        self.infopanel.update_line('blat', 'blat time %5.3f' % np.mean(self.blat_times))
                        self.display.blit(self.infopanel.get_surface(), (2,2))
                    pygame.display.flip()
                    if last_anim_time + 1/30 < st:
                        c_ctr+= animate
                        last_anim_time = st
                else:
                    if logmsgs < 40:
                        print('size mismatch display: %d, %d -> %s' % (win_w, win_h , self.npimg.shape))
                        logmsgs += 1

    def update_lut(self, newsize):
        lut_size = min(newsize, 128)
        if lut_size+1 != self.current_lut.shape[1]:
            self.current_lut = makelookup(lut_size)

    def change_engine(self, key):
        if key == 'o':
            self.engine_vx = (self.engine_vx -1) % len(self.enginelist)
        elif key == 'p':
            self.engine_vx = (self.engine_vx +1) % len(self.enginelist)
        elif key == 'i':
            self.engine_ix = (self.engine_ix -1) % len(self.enginelist[0])
        elif key == 'k':
            self.engine_ix = (self.engine_ix +1) % len(self.enginelist[0])
        self.infopanel.update_line('engine', 'engine: %s' % self.enginelist[self.engine_vx][self.engine_ix][1])
        self.calc_times = []
        self.recalc = True
        return self.enginelist[self.engine_vx][self.engine_ix][0]

    def mouse_to_universe(self, mousepos):
        """
        given a mousepos (2_tuple of x & y within the current screen window), returns the equivalent
        position in the universe
        """
        yv=mousepos[1]/self.winy
        return self.cur_univ.xs.min_v + self.cur_univ.xs.size*mousepos[0]/self.winx, self.cur_univ.ys.min_v + self.cur_univ.ys.size*mousepos[1]/self.winy

def blat(npdata, npimg, lutgroup, offset):
    cols, rows=npdata.shape
    offs, lutno = divmod(offset,4)
    lut = lutgroup[lutno]
    lutlen = len(lut)
    for x in range(cols):
        for y in range(rows):
            npimg[x,y] = lut[(npdata[x,y]+offs) % lutlen]

j_blat = numba.njit(blat)

def addengines(enginelist, desc, descsa, cvers):
    for ix, f in enumerate(cvers):
        if f is None:
            enginelist[ix].append((f, '%9s %s not on this machine' % (desc,descsa[ix])))
        else:
            enginelist[ix].append((f, '%9s %s' % (desc,descsa[ix])))

if __name__ == '__main__':
    import mandcalcs
    if not mandcalcs.withnumba:
        print('Unable tp run, numba or numpy not found (import numba, numpy failed)', file=sys.stderr)
        sys.exit(1)
    enginelist=[[],[],[]]
    descsa=['straight python','njit version','njit parallel']
    for calc, desc in [
        (mandcalcs.m_calc, 'simple'),
        (mandcalcs.m_nsqu_calc, 'no squrt'),
        ]:
        cvers = mandcalcs.get_calc_versions(calc)
        addengines(enginelist, desc, descsa, cvers)
        
    cvers = (mandcalcs.numpy_calc32, mandcalcs.numpy_j_calc32, mandcalcs.numpy_jp_calc32)
    addengines(enginelist, 'numpy float32', descsa, cvers)
    cvers = (mandcalcs.numpy_calc64, mandcalcs.numpy_j_calc64, mandcalcs.numpy_jp_calc64)
    addengines(enginelist, 'numpy float64', descsa, cvers)
    cvers = (mandcalcs.numpy_calc128, mandcalcs.numpy_j_calc128, mandcalcs.numpy_jp_calc128)
    addengines(enginelist, 'numpy float128', descsa, cvers)
    cvers = (mandcalcs.xc_calc, mandcalcs.xc_j_calc, mandcalcs.xc_jp_calc)
    addengines(enginelist, 'vectorized', descsa, cvers)

    fraccer = fracman(enginelist = enginelist)
    fraccer.run_loop()
      
    pygame.quit()
