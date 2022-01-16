
#define image sequence
#define base images as empty squares
#cyle through images calling functions for relevant type
    #pass current image set
    #call functions to analyse each image
    #generate image based on returned data
    #return new image
#analyse image change
#stop loop when change small or set cap on iterations

import matplotlib.pyplot as plt
import cv2
import numpy as np

def BGR (RGB_colour):
    BGR_colour = [RGB_colour[2],RGB_colour[1],RGB_colour[0]]

    return BGR_colour

def analyse_colour_percent(image,colour_filter):
    
    diff = 10

    lower = (BGR(colour_filter))
    upper = (BGR(colour_filter))

    for x in range(0,3):
        lower[x] = max(lower[x]-diff,0)
        upper[x] = min(upper[x]+diff,255)
        
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    imask = cv2.inRange(image,lower,upper)

    ratio_colour = cv2.countNonZero(imask)/(image.size)
    #colourPercent = (ratio_colour * 100)
    #print(ratio_colour)
    #print('black pixel percentage:', np.round(colourPercent, 2))

    return ratio_colour

def pie_chart(black_ratio):
    chart_values = np.array([black_ratio,1-black_ratio])
    chart_labels = ['fraction of \nthis image \nwhich is black','fraction of \nthis image \nwhich is white']
    chart_colours = ['black','white']
    wedgeprops = {'edgecolor':'black','linestyle':'-','linewidth': 2}
    fig, ax = plt.subplots()
    ax.pie(chart_values, colors = chart_colours,wedgeprops = wedgeprops,labels = chart_labels)
    #plt.show()
    fig.canvas.draw()
    #buf = fig.canvas.tostring_rgb()
    #ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    plt.close('all')

    return image

def bar_chart(black_ratios):

    chart_values = np.array(black_ratios)
    x_values = [1,2,3]
    tick_labels = [str(1),str(2),str(3)]
    fig,ax = plt.subplots()
    ax.bar(x_values,chart_values,color = 'black',width=0.4, tick_label=tick_labels)
    ax.axes.yaxis.set_ticklabels([])
    ax.set_title('amount of black \nink by panel:')
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    plt.close('all')
    
    return image

def x_y_chart(base_image):
    
    RGB_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    
    fig,ax =  plt.subplots()
    ax.imshow(RGB_image)

    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.set_title('location of \nblack ink in \nthis image:')

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    plt.close('all')

    return image


def combine_images(image_one,image_two,image_three):

    combined_image = np.concatenate((image_one, image_two,image_three), axis=1)
    return combined_image

def add_border(image):
    border_width = 5
    border_colour = [00, 00, 00]
    output_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=border_colour)
    return output_image


if __name__ == "__main__":

    colour_filter = [00, 00, 00]  # RGB for black

    ratio = 1
    ratio_delta = 1

    bar_ratios = [1,1,1]
    bar_ratio_delta= [1,1,1]
    new_bar_ratios = bar_ratios

    image_three = pie_chart(0) # placeholder for image 3
    image_three = add_border(image_three)

    i=1
    while abs(ratio_delta) > 0.0001:
        image_one = pie_chart(ratio)
        image_one = add_border(image_one)
        new_bar_ratios[0] = analyse_colour_percent(image_one,colour_filter)
        bar_ratio_delta[0] = new_bar_ratios[0]-bar_ratios[0]
        bar_ratios = new_bar_ratios

        image_two = bar_chart(bar_ratios)
        image_two = add_border(image_two)
        new_bar_ratios[1] = analyse_colour_percent(image_two,colour_filter)
        bar_ratio_delta[1] = new_bar_ratios[1]-bar_ratios[1]
        bar_ratios = new_bar_ratios

        image = combine_images(image_one,image_two,image_three)

        image_three = x_y_chart(image)
        image_three = add_border(image_three)        
        new_bar_ratios[2] = analyse_colour_percent(image_three,colour_filter)
        bar_ratio_delta[2] = new_bar_ratios[2]-bar_ratios[2]
        bar_ratios = new_bar_ratios

        image = combine_images(image_one,image_two,image_three)
        new_ratio = analyse_colour_percent(image,colour_filter)
        ratio_delta = new_ratio-ratio
        ratio = new_ratio

        print(np.round(ratio_delta,4))
        print(np.round(bar_ratio_delta,4))

        save_loc = r'output\outfile'+str(i)+'.jpg'
        cv2.imwrite(save_loc,image)
        i=i+1


    




    
    
    






        

