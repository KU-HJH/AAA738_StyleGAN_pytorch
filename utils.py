


def model_summary(generator):
    idx = 0
    # temp = []
    prev = 16
    for s_i, c in enumerate(generator.generator.progression.children()):
    #     temp.append(c)
    # temp = temp[::-1]
    # print(temp[0])
    # for s_i, c in enumerate(temp):
        # print(type(c))
        # for a in c:
        print('\n{} -- {}th'.format(str(c).split('(')[0], s_i + 1))
        print("=" * 80) 
        # for a1 in a.children():
        
        for a1 in c.children():
            if 'model' in str(type(a1)) or 'Sequential' in str(type(a1)):
                # print(a1.children())
                if 'model' in str(type(a1)):
                    print('[ {} ]'.format(str(a1).split('(')[0]))
                for a2 in a1.children(): # designed module
                    if 'EqualLinear' in str(a2) or 'EqualConv2d' in str(a2):
                        toprint = str(a2).split(',')
                        name, *_, in_ = toprint[0].split('(')
                        in_c = in_.replace('in_features=', '')
                        out_c = toprint[1].replace('out_features=', '')
                        name = name.replace(',', '')
                        in_c = in_c.replace(',', '')
                        if 'EqualLinear' in str(a2):
                            if 'AdaptiveInstanceNorm' in str(a1):
                                print('{:>20} {:>25} {:>15} {:>15} {:>15} {:>15}, {}'.format(name, in_c, '-->', out_c, '-->', int(out_c)//2, int(out_c)//2))
                            else:
                                print('{:>20} {:>25} {:>15} {:>15} {:>15} {:>15}, {}'.format(name, in_c, '-->', out_c, '-->', int(out_c)//2, int(out_c)//2))
                                
                            prev = int(out_c) // 2
                        else:
                            print('{:>20} {:>25} {:>15} {:>15} '.format(name, in_c, '-->', out_c))
                            prev = int(out_c)
                    elif 'Conv2d' in str(a2):
                        toprint = str(a2).split('(')
                        name = toprint[0]
                        in_c, out_c = toprint[1].split(',')[:2]
                        name = name.replace(',', '')
                        in_c = in_c.replace(',', '')
                        print('{:>20} {:>25} {:>15} {:>15}'.format(name, in_c, '-->', out_c))
                    elif 'InstanceNorm2d' in str(a2):
                        toprint = str(a2).split('(')
                        name, ch = toprint[0], toprint[1].split(',')[0]
                        print('{:>20} {:>25} {:>15} {:>15}'.format(name, ch, '-->', ch))
                        # print(str(a2))
                    elif 'FusedUpsample' in str(a2) or 'Upsample' in str(a2):
                        if 'FusedUpsample' in str(a2):
                            in_c, out_c , *_ = a2.weight.shape
                            print('{:>20} {:>25} {:>15} {:>15}'.format(str(a2).split('(')[0], in_c, '-->', out_c))
                        else:
                            print('{:>20} {:>36}'.format(str(a2).split('(')[0], 'Keep Dimension'))
                            
                    else:
                        print('{:>20}'.format(str(a2).split('(')[0]))

                    idx += 1
            else: # nn.module
                if 'Conv2d' in str(a1):
                    toprint = str(a1).split('(')
                    name = toprint[0]
                    in_, out, *_ = toprint[1].split(',')[:2]
                    name = name.replace(',', '')
                    in_ = in_c.replace(',', '')
                    print('{:>20} {:>25} {:>15} {:>15}'.format(name, in_, '-->', out))
                else:
                    print('{:>20}'.format(str(a1).split('(')[0]))
                idx += 1
        print("=" * 80) 

                # else:
                #     print(type(a1))
   