def mean(lst):
    sum=0
    for i in lst:
        sum += i
    return sum/len(lst)

def median(lst):
    lst= sorted(lst)
    a = len(lst)
    if a%2==0:
        return float( (lst[int(a/2)] + lst[int(a/2) - 1])/2 )
    else:
        return lst[int(a/2) + 1]

def mode(lst):
    freq = {}

    for i in lst:
        if i in freq:
            freq[i] +=1
        else:
            freq[i] = 1
        mx = max(freq.values())

        return [k for k,v in freq.items() if v==mx]

def sd(x):
    var=0
    for i in range(len(x)):
        var += (x[i]- mean(x))**2/(len(x))

        return  var**0.5

def corr(x,y):
    cov=0

    for i in range(len(x)):
        cov += ((x[i] - mean(x))*(y[i] - mean(y)))/(len(x))

    Sdx = sd(x)
    Sdy = sd(y)

    return cov/(sd(x)*sd(y))
