import numpy as np

# alphabets = "abcdefghijklmnopqrstuvwxyz#"
alphabets = "ab#"

def one_hot_encode(name):
    name = name.lower()
    data = []
    name = name+'#'
    for l in name:
        temp = np.zeros((1,len(alphabets)))
        temp[0][alphabets.index(l)] = 1
        data.append(temp)
    return data

def get_letter(data):
    return alphabets[np.argmax(data)]

def normalize_data(data):
    data  = data - np.mean(data)
    data/=np.max(data)
    return data

def clipping(data):
    data[data>6] = 6
    data[data<-6] = -6
    return data

def data_set():
    alpha = "abcdefghijklmnopqrstuvwxyz"
    with open("IndianNames.csv", encoding="utf8") as f:
        names = []
        for name in f.read().split():        
            good_name = True
            for c in name.lower():
                if c not in alpha:
                    good_name = False

            if good_name:
                names.append(name)
                
    from collections import Counter
    return list(dict(Counter(names)).keys())

def sigmoid(x):
      return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    x = sigmoid(x)
    return x * (1 - x)

def tanh_prime(data):
    data = np.tanh(data)
    return 1-data**2

def softmax(data):
    data = data - np.max(data)
    e = np.exp(data)
    out = e/np.sum(e)
    return out

def cross_entropy(c_output,a_output):
    return -np.sum(a_output * np.log(c_output))

def dsoftmax_cross_entropy(c_output,a_output):
    return c_output - a_output
ltm_pt = 0
stm_pt = 0

input_size = 3
hidden_size = 4


#     input data weights
Wxf = np.random.normal(0,1,(input_size,hidden_size))
Wxi = np.random.normal(0,1,(input_size,hidden_size))
Wxc = np.random.normal(0,1,(input_size,hidden_size))
Wxu = np.random.normal(0,1,(input_size,hidden_size))

#     state weights
Wsf = np.random.normal(0,1,(hidden_size,hidden_size))
Wsi = np.random.normal(0,1,(hidden_size,hidden_size))
Wsc = np.random.normal(0,1,(hidden_size,hidden_size))
Wsu = np.random.normal(0,1,(hidden_size,hidden_size))

# ltm weights
Wru = np.random.normal(0,1,(hidden_size,hidden_size))

# dense layer weights
Wy = np.random.normal(0,1,(hidden_size,input_size))   

def lstm(xt,stm_pt,ltm_pt):

      forget_gate = sigmoid(xt @ Wxf + stm_pt @ Wsf) * ltm_pt
      combine =  np.tanh(xt @ Wxc + stm_pt @ Wsc)
      ignore = sigmoid(xt @ Wxi + stm_pt @ Wsi)
      learn_gate = combine * ignore
      ltm_t = remember_gate = forget_gate + learn_gate
      stm_t = np.tanh(remember_gate @ Wru) *  sigmoid(xt @ Wxu + stm_pt @ Wsu)

      return stm_t, ltm_t
def lstm_prime(xt,stm_pt,ltm_pt):

    f = xt @ Wxf + stm_pt @ Wsf
    dfdXt, dfdWxf, dfdstm_pt, dfdWsf = Wxf, xt, Wsf, stm_pt

    c = xt @ Wxc + stm_pt @ Wsc
    dcdxt, dcdWxc, dcdstm_pt, dcdWsc = Wxc, xt, Wsc, stm_pt

    i = xt @ Wxi + stm_pt @ Wsi
    didXt, didWxi, didstm_pt, didWsi = Wxi, xt, Wsi, stm_pt

    u = xt @ Wxu + stm_pt @ Wsu
    dudXt, dudWxu, dudstm_pt, dudWsu = Wxu, xt, Wsu, stm_pt

    f_a = sigmoid(f)
    df_adf = sigmoid_prime(f)

    c_a = np.tanh(c)
    dc_adc = tanh_prime(c)

    i_a = sigmoid(i)
    di_adi = sigmoid_prime(i)

    u_a = sigmoid(u)
    du_adu = sigmoid_prime(u)
    
    l = c_a * i_a
    dldc_a = i_a
    dldi_a = c_a

    forget_gate = f_a * ltm_pt
    dforget_gatedf_a = ltm_pt
    dforget_gatedltm_pt = f_a

    r = forget_gate + l
    drdl = np.ones_like(forget_gate)
    drdforget_gate = np.ones_like(l)

    ur = r @ Wru
    durdWru = r
    durdr = Wru

    ur_a = np.tanh(ur)
    dur_adur = tanh_prime(ur_a)

    stm_t = ur_a * u_a
    dstm_tdur_a = u_a
    dstm_tdu_a = ur_a

    dldc = dldc_a * dc_adc
    dldi = dldi_a * di_adi
    dforget_gatedf = dforget_gatedf_a * df_adf

    drdstm_pt = 0 

    drdWxc = dcdWxc.T @ (drdl * dldc) 
    drdstm_pt += drdl * dldc * dcdstm_pt
    drdWsc = drdl * dldc * dcdWsc

    drdWxi = didWxi.T @ (drdl * dldi)
    drdstm_pt += drdl * dldi * didstm_pt
    drdWsi = drdl * dldi * didWsi

    drdWxf = dfdWxf.T @ (drdforget_gate * dforget_gatedf)
    drdstm_pt += drdforget_gate * dforget_gatedf * dfdstm_pt
    drdWsf = drdforget_gate * dforget_gatedf * dfdWsf

    drdltm_pt = drdforget_gate * dforget_gatedltm_pt

    dstm_tdur = dstm_tdur_a * dur_adur

    dstm_tdr = dstm_tdur * durdr

    dstm_tdWxc = drdWxc @ dstm_tdr 
    dstm_tdWsc = dstm_tdr * drdWsc 

    dstm_tdWxi = drdWxi @ dstm_tdr 
    dstm_tdWsi = dstm_tdr * drdWsi 

    dstm_tdWxf = drdWxf @ dstm_tdr 
    dstm_tdWsf = dstm_tdr * drdWsf

    dstm_tdstm_pt = dstm_tdr * drdstm_pt

    dstm_tdWru = dstm_tdur * durdWru

    dstm_tdltm_pt = dstm_tdr * drdltm_pt
    
    dstm_tdWsu = dudWsu.T @  (dstm_tdu_a * du_adu)
    dstm_tdWxu = dudWxu.T @ (dstm_tdu_a * du_adu)
    dstm_tdstm_pt += dstm_tdu_a * du_adu * dudstm_pt

    dstm_t = [ dstm_tdWxf, dstm_tdWxi, dstm_tdWxc, dstm_tdWxu, dstm_tdWsf, dstm_tdWsi, dstm_tdWsc, dstm_tdWsu, dstm_tdWru, dstm_tdstm_pt, dstm_tdltm_pt]

    dltm_t = [ drdWxf, drdWxi, drdWxc, drdWsf, drdWsi, drdWsc, drdstm_pt, drdltm_pt]
    

    return dstm_t,dltm_t
	
	def train(name):

    xt = one_hot_encode(name)[:-1]
    yt = one_hot_encode(name)[1:]
    error = 0
    
    xs,ys,yhats,stm_ts,stm_pts,ltm_ts,ltm_pts = [],[],[],[],[],[],[]
    stm_pt,ltm_pt = np.zeros((1,hidden_size)), np.zeros((1,hidden_size))
    for i in range(len(xt)):

        x,y = xt[i],yt[i]
        stm_t, ltm_t= lstm(x,stm_pt,ltm_pt)
        yhat= softmax(stm_t @ Wy)

        xs.append(x)
        ys.append(y)
        yhats.append(yhat)
        stm_ts.append(stm_t)
        stm_pts.append(stm_pt)
        ltm_ts.append(ltm_t)
        ltm_pts.append(ltm_pt)

        # print(x,y,yhat,stm_t,stm_pt,ltm_t,ltm_pt)
        stm_pt, ltm_pt= stm_t,ltm_t
        # print(get_letter(yhat),end='')

    dWxf  = np.zeros((input_size,hidden_size)),
    dWxi  = np.zeros((input_size,hidden_size)),
    dWxc  = np.zeros((input_size,hidden_size)),
    dWxu  = np.zeros((input_size,hidden_size)),
  
    dWsf  = np.zeros((hidden_size,hidden_size)),
    dWsi  = np.zeros((hidden_size,hidden_size)),
    dWsc  = np.zeros((hidden_size,hidden_size)),
    dWsu  = np.zeros((hidden_size,hidden_size)),
  
    dWru  = np.zeros((hidden_size,hidden_size)),
  
    dWy  = np.zeros((hidden_size,input_size))   

    for i in range(len(xt)):

        stm_t = stm_pts[i]
        yt = ys[i]
        yhat = yhats[i]

        dEdy = dsoftmax_cross_entropy(yhat,yt)

        dWy += stm_t.T @ dEdy

        dEdstm_t = dEdy @ Wy.T

        dEdstm_pt = None
        dEdltm_pt = None
        
        for j in range(i, -1, -1):

            dstm_td,dltm_td = lstm_prime(xs[i],stm_pts[i],ltm_pts[i])
            dstm_tdWxf, dstm_tdWxi, dstm_tdWxc, dstm_tdWxu, dstm_tdWsf, dstm_tdWsi, dstm_tdWsc, dstm_tdWsu, dstm_tdWru, dstm_tdstm_pt, dstm_tdltm_pt = dstm_td
            dltm_tdWxf, dltm_tdWxi, dltm_tdWxc, dltm_tdWsf, dltm_tdWsi, dltm_tdWsc, dltm_tdstm_pt, dltm_tdltm_pt = dltm_td
            # print(dltm_tdltm_pt.shape)
            if i == j:
                 dWxf = dstm_tdWxf * dEdstm_t 
                 dWxi = dstm_tdWxi * dEdstm_t
                 dWxc = dstm_tdWxc * dEdstm_t
                 dWxu = dstm_tdWxu * dEdstm_t
                 dWsf = dstm_tdWsf * dEdstm_t
                 dWsi = dstm_tdWsi * dEdstm_t
                 dWsc = dstm_tdWsc * dEdstm_t
                 dWsu = dstm_tdWsu * dEdstm_t
                 dWru = dstm_tdWru * dEdstm_t
                 dEdstm_pt =  dstm_tdstm_pt * dEdstm_t
                 dEdltm_pt = dstm_tdltm_pt * dEdstm_t
            else:
                 dWxf   += (dstm_tdWxf + dltm_tdWxf) * dEdstm_pt
                 dWxi   += (dstm_tdWxi + dltm_tdWxi) * dEdstm_pt
                 dWxc   += (dstm_tdWxc + dltm_tdWxc) * dEdstm_pt
                 dWsf   += (dstm_tdWsf + dltm_tdWsf) * dEdstm_pt
                 dWsi   += (dstm_tdWsi + dltm_tdWsi) * dEdstm_pt
                 dWsc   += (dstm_tdWsc + dltm_tdWsc) * dEdstm_pt
                 dWxu   += dstm_tdWxu * dEdstm_pt
                 dWsu   += dstm_tdWsu * dEdstm_pt
                 dWru   += dstm_tdWru * dEdstm_pt
                 dEdstm_pt *=  dstm_tdstm_pt + dltm_tdstm_pt
                 dEdltm_pt *= dstm_tdltm_pt + dltm_tdltm_pt

    learning_rate = 0.001
    Wxf += learning_rate * dWxf
    Wxi += learning_rate * dWxi
    Wxc += learning_rate * dWxc
    Wsf += learning_rate * dWsf
    Wsi += learning_rate * dWsi
    Wsc += learning_rate * dWsc
    Wxu += learning_rate * dWxu
    Wsu += learning_rate * dWsu
    Wru += learning_rate * dWru
		
train("abba")
    
