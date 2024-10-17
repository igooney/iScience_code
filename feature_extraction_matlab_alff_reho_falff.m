%读取文件列表 ALFF
Alff_path = 'D:\核磁与机器学习2\核磁与机器学习\part5-6\fMRI_features\ALFF_FunImgARCW'
Alff_path_list = dir(Alff_path)
Alff_path_list(1:2) = []

mask = spm_vol('D:\核磁与机器学习2\核磁与机器学习\part5-6\AAL_61x73x61_YCG.nii');
mdata = spm_read_vols(mask);

% fmri = spm_vol('test.nii');
% fdata = spm_read_vols(fmri);
% size(fdata)
alff_data = zeros([871,116]);
for i=1:1:size(Alff_path_list,1)
    i
    path = [Alff_path,'\',Alff_path_list(i).name]
    fmri = spm_vol(path);
    fdata = spm_read_vols(fmri);
    time = size(fdata,4);
    fdataTran = reshape(fdata,[],time);
    mdata = reshape(mdata,[],1); % 将mask转换为一个列向量，当成索引表
    slide116 = zeros(116,time);
    for j = 1:116;
        roi_index = find(mdata==j);
        slide116(j,:) = mean(fdataTran(roi_index,:));
    end
    alff_data(i,:) = slide116';
end


%读取文件列表 fALFF
fAlff_path = 'C:\Users\lwh\Desktop\核磁与机器学习\fMRI_features\fALFF_FunImgARCW'
fAlff_path_list = dir(fAlff_path)
fAlff_path_list(1:2) = []

mask = spm_vol('AAL_61x73x61_YCG.nii');
mdata = spm_read_vols(mask);

% fmri = spm_vol('test.nii');
% fdata = spm_read_vols(fmri);
% size(fdata)
fAlff_data = zeros([871,116]);
for i=1:1:size(fAlff_path_list,1)
    i
    path = [fAlff_path,'\',fAlff_path_list(i).name]
    fmri = spm_vol(path);
    fdata = spm_read_vols(fmri);
    time = size(fdata,4);
    fdataTran = reshape(fdata,[],time);
    mdata = reshape(mdata,[],1); % 将mask转换为一个列向量，当成索引表
    slide116 = zeros(116,time);
    for j = 1:116;
        roi_index = find(mdata==j);
        slide116(j,:) = mean(fdataTran(roi_index,:));
    end
    fAlff_data(i,:) = slide116';
end


%读取文件列表 ReHo
ReHo_path = 'C:\Users\lwh\Desktop\核磁与机器学习\fMRI_features\ReHo_FunImgARCWF'
ReHo_path_list = dir(ReHo_path)
ReHo_path_list(1:2) = []

mask = spm_vol('AAL_61x73x61_YCG.nii');
mdata = spm_read_vols(mask);

% fmri = spm_vol('test.nii');
% fdata = spm_read_vols(fmri);
% size(fdata)
ReHo_data = zeros([871,116]);
for i=1:1:size(ReHo_path_list,1)
    i
    path = [ReHo_path,'\',ReHo_path_list(i).name]
    fmri = spm_vol(path);
    fdata = spm_read_vols(fmri);
    time = size(fdata,4);
    fdataTran = reshape(fdata,[],time);
    mdata = reshape(mdata,[],1); % 将mask转换为一个列向量，当成索引表
    slide116 = zeros(116,time);
    for j = 1:116;
        roi_index = find(mdata==j);
        slide116(j,:) = mean(fdataTran(roi_index,:));
    end
    ReHo_data(i,:) = slide116';
end




