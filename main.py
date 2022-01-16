
import matplotlib.pyplot as plt
import cv2
import numpy as np

def BGR (RGB_colour):
    #convert RGB to BGR
    BGR_colour = [RGB_colour[2],RGB_colour[1],RGB_colour[0]]

    return BGR_colour

def analyse_colour_ratio(image,colour_filter):
    #based on BGR colour definition
    diff = 10 #modify for match strictness

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
    #print('colour pixel percentage:', np.round(colourPercent, 2))

    return ratio_colour

def pie_chart(black_ratio):
    chart_values = np.array([black_ratio,1-black_ratio])
    chart_labels = ['fraction of \nthis image \nwhich is black','fraction of \nthis image \nwhich is white']
    chart_colours = ['black','white']
    wedgeprops = {'edgecolor':'black','linestyle':'-','linewidth': 2}
    fig, ax = plt.subplots()
    ax.pie(chart_values, colors = chart_colours,wedgeprops = wedgeprops,labels = chart_labels)

    fig.canvas.draw()
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

    #assumes images have same height, would need to be more robust for more varied input images
    combined_image = np.concatenate((image_one, image_two,image_three), axis=1)
    return combined_image

def add_border(image):
    #add black border 5 wide
    border_width = 5
    border_colour = [00, 00, 00]
    output_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=border_colour)

    #add white border
    border_colour = [255, 255, 255]
    output_image = cv2.copyMakeBorder(output_image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=border_colour)

    return output_image


if __name__ == "__main__":

    colour_filter = [00, 00, 00]  # RGB for black

    #initialisation values for first charts
    ratio = 1
    bar_ratios = [1,1,1]

    ratio_delta = 1
    bar_ratio_delta= [1,1,1]
    new_bar_ratios = bar_ratios

    image_three = pie_chart(0) # placeholder for image 3 as it is self referential and first version needs a placeholder to function.
    image_three = add_border(image_three)

    i=1
    while abs(ratio_delta) > 0.00001: #stop when ratio change is less than 0.001% Note that visible pie chart in panel 3 is effectively 3 iterations old so relatively strict threshold required
        image_one = pie_chart(ratio)
        image_one = add_border(image_one)
        new_bar_ratios[0] = analyse_colour_ratio(image_one,colour_filter)
        bar_ratio_delta[0] = new_bar_ratios[0]-bar_ratios[0]
        bar_ratios = new_bar_ratios

        image_two = bar_chart(bar_ratios)
        image_two = add_border(image_two)
        new_bar_ratios[1] = analyse_colour_ratio(image_two,colour_filter)
        bar_ratio_delta[1] = new_bar_ratios[1]-bar_ratios[1]
        bar_ratios = new_bar_ratios

        image = combine_images(image_one,image_two,image_three)

        image_three = x_y_chart(image)
        image_three = add_border(image_three)        
        new_bar_ratios[2] = analyse_colour_ratio(image_three,colour_filter)
        bar_ratio_delta[2] = new_bar_ratios[2]-bar_ratios[2]
        bar_ratios = new_bar_ratios

        image = combine_images(image_one,image_two,image_three)
        new_ratio = analyse_colour_ratio(image,colour_filter)
        ratio_delta = new_ratio-ratio
        ratio = new_ratio

        print(np.round(ratio_delta,4))
        #print(np.round(bar_ratio_delta,4))

        save_loc = r'output\outfile'+str(i)+'.jpg'
        cv2.imwrite(save_loc,image)
        i=i+1

    save_loc = r'output\final.jpg'    
    cv2.imwrite(save_loc,image)


    




    
    
    






        

