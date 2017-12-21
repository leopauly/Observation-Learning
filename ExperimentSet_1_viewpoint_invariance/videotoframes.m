%Program for splitting a video into frames 
%Author: @leopauly

clc
clear all
close all

%fileList = dir *.avi;
%L=length(filelist);
%vid = VideoReader([filelist(seq_num).name,'.avi']);

vid = VideoReader('./Sermanet Pouring Dataset/videos/test/box_to_white1_real_view1.mov');
num_frames = vid.NumberOfFrames;
step_size=5;
re_size=[112,112];
frame_num=0;
angle=270
 for i = 1:step_size:num_frames
 frames = read(vid,i);
 frames=imresize(frames,re_size);
 frames =imrotate(frames,angle,'bilinear','crop'); % Rotating
 imwrite(frames,['./Viewinvariance_exp_Dataset/pour0_view1/',int2str(frame_num),'.png']);
 frame_num=frame_num+1;
 im(i)=image(frames);
 i
 end