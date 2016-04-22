slist = list()
f = open("schrauben.txt", "r")
for line in f:
    kind = line[0]
    size = int(line[2:-1])
    slist.append([kind, size])
f.close()
slist.sort()

res = list()
oldkind = None
oldsize = None
amount = 0
i = iter(slist)
for item in i:
    if (item[0] != oldkind) or (item[1] != oldsize):
        if oldkind != None:
            res.append([oldkind, oldsize, amount])
        oldkind, oldsize, amount = item[0], item[1], 1
    else:
        amount = amount + 1

ress = list()
for l in res:
    if l[0] == 'm':
        for ll in res:
            if ll[0] == 's' and ll[1] == l[1]:
                ress.append([l[1], min(ll[2], l[2])])

r = open("Loesung", "w")
r.write("{0:10}     {1:10}\n\n".format("mm", "Paare"))
for l in ress:
    r.write("{0:10} --> {1:10d}\n".format(l[0], l[1]))
					
                    

     
