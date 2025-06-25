import numpy as np
import matplotlib.pyplot as plt
import math

def lambert(r1v, r2v, mu, TOF):
    def F(a, b, c, x):
        return 1 + (a*b/c)*x + (a*(a+1)*b*(b+1)/(c*(c+1)))*(x**2/2) + (a*(a+1)*(a+2)*b*(b+1)*(b+2)/(c*(c+1)*(c+2)))*(x**3/6)

    def Y(x, l):
        return math.sqrt(1-l*l*(1-x*x))

    def N(x, l):
        return Y(x, l) - l*x

    def S(x, l):
        return (1-l-x*N(x, l))/2

    def Q(x, l):
        return 4*F(3, 1, 5/2, S(x, l))/3

    def T(x, l):
        if (x > 0.9) & (x < 1.1):
            return (N(x, l)**3*Q(x, l)+4*l*N(x, l))/2
        else:
            y = Y(x, l)
            if x < 1:
                psi = math.acos(x*y+l*(1-x*x))
            else:
                psi = math.acosh(x*y-l*(x*x-1))
            return (1/(1-x*x))*(psi/math.sqrt(abs(1-x*x))-x+l*y)
    
    def dT(x, l, T):
        return (3*T*x-2+2*l**3*x/Y(x, l))/(1-x**2)

    def d2T(x, l, T, dT):
        return (3*T+5*x*dT+2*(1-l**2)*l**3/(Y(x, l)**3))/(1-x**2)

    def d3T(x, l, T, dT, d2T):
        return (7*x*d2T+8*dT-6*(1-l**2)*l**5*x/(Y(x, l)**5))/(1-x**2)     

    if np.linalg.norm(np.cross(r1v, r2v)) == 0:
        a = (np.linalg.norm(r1v)+np.linalg.norm(r2v))/2
        r1 = np.linalg.norm(r1v)
        r2 = np.linalg.norm(r2v)
        ra = max(r1, r2)
        e = ra/a-1
        p = a*(1-e**2)
        h = math.sqrt(mu*p)
        e_h = np.array([0, 0, 1])
        v1v = np.cross(e_h, r1v/r1)
        v2v = np.cross(e_h, r2v/r2)
        v1 = (h/r1)*v1v
        v2 = (h/r2)*v2v
    else:
        r1 = np.linalg.norm(r1v)
        r2 = np.linalg.norm(r2v)
        c = math.sqrt(r1**2+r2**2-2*np.dot(r1v, r2v))
        s = (r1+r2+c)/2

        g = math.sqrt(s*mu/2)
        p = (r1-r2)/c
        o = math.sqrt(1-p**2)

        e_r1 = r1v/r1
        e_r2 = r2v/r2
        e_h = np.cross(e_r1, e_r2)/(np.linalg.norm(np.cross(e_r1, e_r2)))
        if (r1v[0]*r2v[1]-r1v[1]*r2v[0]) < 0:
            e_t1 = np.cross(e_r1, e_h)
            e_t2 = np.cross(e_r2, e_h)
            l = -math.sqrt(1-c/s)
        else:
            e_t1 = np.cross(e_h, e_r1)
            e_t2 = np.cross(e_h, e_r2)
            l = math.sqrt(1-c/s)

        Td = math.sqrt(2*mu/s**3)*TOF
        T1 = 2*(1-l**3)/3
        T0 = math.acos(l) + l*math.sqrt(1-l**2)

        if Td < T1:
            x0 = (T1/Td)-1
        elif Td < T0:
            x0 = pow(T0/Td, math.log2(T1/T0))-1
        else:
            x0 = pow(T0/Td, 2/3)-1

        x_ite = np.linspace(0, 0, 10)

        for i in range(3):
            Ts = T(x0, l)
            dTs = dT(x0, l, Ts)
            d2Ts = d2T(x0, l, Ts, dTs)
            Ts = Ts - Td
            x1 = x0 - (Ts*dTs)/(dTs**2-Ts*d2Ts/2)
            x_ite[i] = x1
            x0 = x1

        y1 = Y(x1, l)

        v_r1 = g*((l*y1-x1)-p*(l*y1+x1))/r1
        v_r2 = -g*((l*y1-x1)+p*(l*y1+x1))/r2
        v_t1 = g*o*(y1+l*x1)/r1
        v_t2 = g*o*(y1+l*x1)/r2

        v1 = v_r1*e_r1 + v_t1*e_t1
        v2 = v_r2*e_r2 + v_t2*e_t2

    return v1, v2

def plotOrbit(r1, v1, mu, res, c):
    hv = np.cross(r1, v1)
    E = np.linalg.norm(v1)**2/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    ev = np.cross(v1, hv)/mu - r1/np.linalg.norm(r1)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * (-periapsis[1]/abs(periapsis[1]))
    n = res
    theta = np.linspace(0, 2*math.pi+0.01, num=n)
    x = np.linspace(0, 0, num=n)
    y = np.linspace(0, 0, num=n)
    for i, t in np.ndenumerate(theta):
        r = p/(1+e*math.cos(t))
        x[i] = r*math.cos(t)*math.cos(w)+r*math.sin(t)*math.sin(w)
        y[i] =-r*math.cos(t)*math.sin(w)+r*math.sin(t)*math.cos(w)
    plt.plot(x, y, color=c)

def plotSolution(r1, r2, v1, mu, res, c):
    hv = np.cross(r1, v1)
    E = np.linalg.norm(v1)**2/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    ev = np.cross(v1, hv)/mu - r1/np.linalg.norm(r1)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * (-periapsis[1]/abs(periapsis[1]))
    n = res
    if np.linalg.norm(ev) < 1e-12:
        w = 0
        t1 = 0
        t2 = math.acos(np.dot(r1,r2)/(np.linalg.norm(r1)**2))
    else:
        t1 = math.acos((p-np.linalg.norm(r1))/(np.linalg.norm(r1)*e))
        if np.dot(np.cross(r1,periapsis), np.cross(r1, v1))>0:
            t1 = 2*math.pi - t1
        t2 = math.acos((p-np.linalg.norm(r2))/(np.linalg.norm(r2)*e))
        if np.dot(np.cross(r2,periapsis), np.cross(r1, v1))>0:
            t2 = 2*math.pi - t2
        if t1 > t2:
            t1 = t1 - 2*math.pi
    theta = np.linspace(t1, t2, num=n)
    x = np.linspace(0, 0, num=n)
    y = np.linspace(0, 0, num=n)
    for i, t in np.ndenumerate(theta):
        r = p/(1+e*math.cos(t))
        x[i] = r*math.cos(t)*math.cos(w)+r*math.sin(t)*math.sin(w)
        y[i] =-r*math.cos(t)*math.sin(w)+r*math.sin(t)*math.cos(w)
    plt.plot(x, y, color=c)