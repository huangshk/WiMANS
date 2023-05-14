%
%%
clear;
clc;
close all;
%
%%
var_path = input("Please input the directory of dat files: ", "s");
%
var_path_dat = dir(var_path + "*.dat");
disp("Number of *.dat files: "+ length(var_path_dat))
%
%%
for var_i = 1:length(var_path_dat)
    % if var_i <= 8472
        % continue
    % end
    %
    var_file_dat = var_path_dat(var_i).folder + "\" + var_path_dat(var_i).name;
    trace = read_bf_file(var_file_dat);
    %
    var_file_mat = erase(var_file_dat, ".dat") + ".mat";
    disp(var_file_mat + " - Shape " + length(trace) + " " + mat2str(size(trace{1}.csi)))
    save(var_file_mat, "trace");
end