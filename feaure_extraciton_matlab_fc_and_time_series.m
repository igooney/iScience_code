clc
clear all
mask = spm_vol('D:\机器学习\part5-6\AAL_61x73x61_YCG.nii');
mdata = spm_read_vols(mask);

file_list_path = 'D:\核磁与机器学习\fMRI'
file_list = dir(file_list_path)
file_list = file_list(3:end)

FC_ALL = zeros(size(file_list, 1),116,116);
slide1116_all = zeros(size(file_list, 1), 116,78);
for i  = 1 : 1 : size(file_list, 1)
    fmri_path = [file_list_path,'\' ,file_list(i).name]
    fmri = spm_vol(fmri_path);
    fdata = spm_read_vols(fmri);

    time = size(fdata,4);
    fdataTran = reshape(fdata,[],time);
    mdata = reshape(mdata,[],1); % 将mask转换为一个列向量，当成索引表
    slide116 = zeros(116,time);
    for j = 1:116;
        roi_index = find(mdata==j);
        slide116(j,:) = mean(fdataTran(roi_index,:));
    end
    slide116';
    slide1116_all(i,:,:) =  slide116;
    FC = corr(slide116');
    FC = triu(FC); %取数据上三角
    FC_ALL(i,:,:) = FC;
end



save('D:\核磁与机器学习2\核磁与机器学习\part5-6\time_series.mat', 'slide1116_all');
save('D:\核磁与机器学习2\核磁与机器学习\part5-6\aal_all.mat', 'FC_ALL');



