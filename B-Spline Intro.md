
# Getting to know B-Splines (basis splines)

Splines are piecewise polynomials that are continuously differentiable to one order less than the polynomial that defines it.  For example, a spline that is described by a third order polynomial must be twice differentiable.  The place where these polnomials join are the knots.  I will only be studying uniformly spaced knots.  

Formal definition of B-Splines

$b^n(x) = \int_{x-1}^{x} b^{n-1}$

$\frac{d}{dx}b^{n}(x) = b^{n-1}(x) - b^{n-1}(x-1)$

$b^n(0) = 0$

The first two b-splines are plotted below.


```python
import numpy as np
from matplotlib import pyplot as plt
```


```python
def b1(x):
    return np.piecewise(x, [x>=0 and x<=1, x>=1 and x<=2], [x, 2-x])

def b2(x):
    if(x > 0 and x < 1):
        return 0.5 * x**2
    elif(x >= 1 and x <=2 ):
        return -x**2 + 3.0 * x - 1.5
    elif(x > 2 and x <= 3):
        return  4.5 - 3.0* x +0.5 *x**2
    else:
        return 0

x = np.linspace(-4,4)
bs1 = x.copy()*0.0
bs2 = x.copy()*0.0
for i in range(len(x)):
    bs1[i] = b1(x[i])
    bs2[i] = b2(x[i])

l1, = plt.plot(x,bs1,label='b1')
l2, = plt.plot(x,bs2,label='b2')
plt.legend([l1,l2])
plt.show()
```


![png](B-Spline%20Intro_files/B-Spline%20Intro_2_0.png)


A simpler computation is to use the recurrence relationship proved by de Boor and Cox. Note that this is continuous relationship and not discrete.


```python
def b2(x, n=2.0):
    return x/n*b1(x) + (n+1.0-x)/n * b1(x-1.0)

x= np.linspace(0,4)
    
l1, = plt.plot(x,[b1(y) for y in x],label='b1')
l2, = plt.plot(x,[b2(y) for y in x],label='b2')

plt.legend([l1,l2])
plt.show()
        
```


![png](B-Spline%20Intro_files/B-Spline%20Intro_4_0.png)


A more efficient way to construt the B-Splines is to use the recurrance relationship to derive coefficients for the Taylor series representation of the polynomials.  Here we use coefficients $a_{k,l}^{n}$ where $n$ is the B-Spline order, $k$ is the interval on the abscicca and $l$ is the Taylor series term. "


```python
NumPolys=5
def TaylorCoefficients(N):
    a = np.zeros([N+1,N+1,N+1])
    a[0,0,0] = 1.0
    b = np.zeros(4)
    for n in range(1,N+1):
        for k in range (0,N+1):
            for l in range(0,N+1):
                b[0] = a[n-1,k,l]
                if (l-1 < 0):
                    b[1] = 0.0
                else:
                    b[1] = a[n-1,k,l-1]
                if(k-1 < 0):
                    b[2] = 0
                else:
                    b[2] = a[n-1,k-1,l]
                if(k-1 < 0 or l-1 <0):
                    b[3] = 0
                else:
                    b[3] = a[n-1,k-1,l-1]
                a[n,k,l] = float(k)/n * b[0] \
                         + 1.0/n * b[1] \
                         + (n+1.0-k)/n * b[2] \
                         - 1.0/n * b[3]
    return a
#a = TaylorCoefficients(NumPolys)
#print(a)

def BSplinePlot(N, xn=20):    
    a = TaylorCoefficients(N)
    x = np.linspace(0,N+1, xn*(N+1))
    p = np.zeros(xn*(N+1))
    for k in range(N+1):
        for l in range(N+1):
            p[k*xn:(k+1)*xn] += a[N,k,l]*(x[k*xn:xn*(k+1)]-k)** l
    return x,p
h=[]
for i in range(1,NumPolys+1):
    x,p = BSplinePlot(i)
    line,=plt.plot(x,p,label='n={}'.format(i))
    h.append(line)
plt.legend(h)
plt.show()
```


![png](B-Spline%20Intro_files/B-Spline%20Intro_6_0.png)


The b-splines can also be scaled and translated similar to wavelets.  For this the identity is $b_{k,h}^n = b^n(x/h-k)$ for $h>0$ and $k \in \mathcal{Z}$


```python
def ScaledTranslatedBSpline(H,K,N,xn=20):
    a = TaylorCoefficients(N)
    x = np.linspace(0,N+1, xn*(N+1))*H+K*H
    p = np.zeros(xn*(N+1))
    for k in range(N+1):
        for l in range(N+1):
            p[k*xn:(k+1)*xn] += a[N,k,l]*(x[xn*k:xn*(k+1)]/H-K-k)** l
    return x,p
def BSplineValueAt(x,H,N,K):
    a = TaylorCoefficients(N)
    p = x * 0.0
    for k in range(N+1):
        for l in range(N+1):
            ptemp = a[N,k,l]*(x/H-K-k)** l
            ptemp[x<(k+K)*H]=0.0
            ptemp[x>=(k+K+1)*H]=0.0
            #print(ptemp)
            p+=ptemp
    return p
h = 0.5
N = 3
handles = []
for k in range(-3,2):
    x,p = ScaledTranslatedBSpline(h,k,N)
    line,=plt.plot(x,p,label="h={} k={} N={}".format(h,k,N))
    handles.append(line)
#plt.legend(handles) 
plt.show()
```


![png](B-Spline%20Intro_files/B-Spline%20Intro_8_0.png)



```python
# show that the sum of all un-scaled b-splines 
# with support on an interval sum to 1  
# this is a carndial spline and the space spanned by the 
# basis splines
x=np.linspace(-3.0,5.0,200)
psum = x * 0.0
h = 0.5
N = 3

for k in range(-5,5):
    plt.plot(x,BSplineValueAt(x,h,N,k),'-',label="k={}".format(k))
    psum += BSplineValueAt(x,h,N,k)
plt.plot(x,psum,'-o')
#plt.legend()
plt.show()
        
```


![png](B-Spline%20Intro_files/B-Spline%20Intro_9_0.png)



```python
# Marsden's identity
# Showing that polynomials can be repesented as cardinal splines 
# by multiplying the individual b-splines by weights

def psi(N,K,H,T):
    a = H**N
    for n in range(1,N+1):
        a *= K+n-t/H
    return a

n=2
h=1.0
t=0.0
x = np.linspace(-10,10,500)
psum = x*0.0

for k in range(-8,6):
    plt.plot(x,psi(n,k,h,t)*BSplineValueAt(x,h,n,k),'-',label="k={}".format(k))
    psum += psi(n,k,h,t)*BSplineValueAt(x,h,n,k)
plt.plot(x,psum,'-k',x,x**2,'--r')
#plt.legend()
plt.show()
    
```


![png](B-Spline%20Intro_files/B-Spline%20Intro_10_0.png)


# Subdivision 
A b-spline can be made up of a weighted combination of b-splines of the same order on a reduced grid size.  This is done in the following formula:

$b_{k,h}^n = 2^{-n} \sum_{l=0}^{n+1} bi_{n+1,l} b_{2k+l,h/2}^n$

where $bi_{n+1,l}$ are the binomial coefficients 

$\frac{(n+1)!}{l!((n+1-l)!}$

Rather than computing the binomial coefficients directly we can use an averaging procedure to compute the coefficients $c'_{2k+l}=2^{-n} bi_{n+1,l}$.  This is done by initializing the coefficients $c'_{2k}$ and $c'_{2k+1}$ to 1, and setting all others to zero. Then the averaging 

$c'_l \leftarrow 1/2 (c'_l + c'_{l-1})$

is perfomed n times.  The convention for both cases is the coefficient is zero if $l<0$ or $l>n+1$.

Their equivalence is shown below




```python
from math import factorial
    
def binomial_coefficient(k,l):
    return factorial(k)/(factorial(l)*factorial(k-l))

def c_kl(n):
    c_l = np.zeros(n+3)
    c_l[1:3]=1.0
    for i in range(n):
        temp=c_l.copy()
        c_l[1:i+4]+=temp[0:i+3]
        c_l*=0.5
    return c_l[1::]

for n in range(1,5):
    l = np.arange(n+2)
    b = np.array([binomial_coefficient(n+1,ll) for ll in l]) * 2.0**-n
    c = c_kl(n)
    print("n={}".format(n))
    print("b coefficients",b)
    print("c coefficients",c)
    
x= np.linspace(0,4,200)
h1 = 1.0
h2 = 0.5
n=3

c=c_kl(n)
b_h1 = BSplineValueAt(x,h1,n,0)


b_card=b_h1*0.0
for k in range(len(c)):
    b = c[k]*BSplineValueAt(x,h2,n,k)
    b_card += b
    plt.plot(x,b,'--k')
plt.plot(x,b_card,'ok')
plt.plot(x,b_h1)
plt.show()
    
```

    n=1
    b coefficients [ 0.5  1.   0.5]
    c coefficients [ 0.5  1.   0.5]
    n=2
    b coefficients [ 0.25  0.75  0.75  0.25]
    c coefficients [ 0.25  0.75  0.75  0.25]
    n=3
    b coefficients [ 0.125  0.5    0.75   0.5    0.125]
    c coefficients [ 0.125  0.5    0.75   0.5    0.125]
    n=4
    b coefficients [ 0.0625  0.3125  0.625   0.625   0.3125  0.0625]
    c coefficients [ 0.0625  0.3125  0.625   0.625   0.3125  0.0625]



![png](B-Spline%20Intro_files/B-Spline%20Intro_12_1.png)



```python
# Scalar product and scalar product of derivatives

def s(n,k,l,h):
    return 
```
