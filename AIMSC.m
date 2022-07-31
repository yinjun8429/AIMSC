%% 基于锚(anchor)的不完全多视角谱聚类
%% 采用交替迭代法求解样本和锚点之间的相似度矩阵W，要求W的元素均大于0,使用二次规划求解
%% maxiter为最大迭代次数
%% stoperr为判断迭代结束的误差
%% alpha为规则化参数
%% J为最终输出的目标函数值
%% los_mark为每个视角中缺失数据样本的编号
%% data_e为存在数据样本，ori_W是初始的W,data_num为总的数据样本数（包括缺失样本）
%% alpha的设定有技巧
%% 将所有参数统一于结构体options中
%% 最优化目标： min{Xv,W}SUMv=1~V||Xv-Av*W||^2+alpha||W||^2  s.t.Xv*SEv = XEv, W'1=1,  W>=0(1为元素均为1的列向量)
%%
function [data,W,new_data,J] = AIMSC(data_e,anc_data,ori_W,los_mark,options)

alpha = options.alpha;
maxiter = options.maxiter;
stoperr = options.stoperr;
new_dim = options.new_dim;

view_num = length(data_e);
[anc_num,data_num] = size(ori_W); %%anc_num为锚点的个数
dim = zeros(1,view_num); %记录每个视角维数的向量
tq_l = cell(1,view_num);  %缺失提取矩阵 
% tq_e = cell(1,view_num);  %存在提取矩阵
kz_l = cell(1,view_num); %缺失扩展矩阵
kz_e = cell(1,view_num); %存在扩展矩阵
% W = cell(1,view_num); %每个视角样本和锚点间的相似矩阵
tot_mark = 1:data_num;   %总数据标签
% data_e = cell(1,view_num); %每个视角的存在样本矩阵
% data_l = cell(1,view_num);  %每个视角的缺失样本矩阵
data = cell(1,view_num);  %每个视角的完整样本矩阵

sum_dim = 0;

%% 二次规划的输入参数
f = zeros(anc_num,1);
A = [];
b = [];
Aeq = ones(1,anc_num);
beq = 1;
lb = zeros(anc_num,1);
ub = [];
%%

for view_mark = 1:view_num    
    [dim(view_mark),~] = size(data_e{view_mark});
    sum_dim = sum_dim+dim(view_mark);
    los_mark{view_mark} = sort(los_mark{view_mark});  %将缺失样本的标签按顺序排列
%      tq_e{view_mark} = eye(data_num);
%      tq_e{view_mark}(:,los_mark{view_mark}) = [];
     ext_mark = setdiff(tot_mark,los_mark{view_mark});
     tq_l{view_mark} = eye(data_num);
     tq_l{view_mark}(:,ext_mark) = [];
     l_num = length(los_mark{view_mark}); %每个视角缺失样本的个数
     e_num = data_num - l_num;  %每个视角存在样本的个数
     kz_l{view_mark} = zeros(l_num,data_num);
     kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
     kz_e{view_mark} = zeros(e_num,data_num);
     kz_e{view_mark}(:,ext_mark) = eye(e_num);
%     data_e{view_mark} = data_t{view_mark}*tq_e{view_mark};
end
alpha = alpha*sum_dim/anc_num;

%%%%将相似度矩阵设为初始值
W = ori_W;  
%%%%

%% 迭代求解相似度矩阵W和完整样本矩阵data
J = zeros(maxiter,1);
for it_mark = 1:maxiter   %%迭代求解
     %%%%更新完整样本矩阵      此处对应于论文中的 X(v)=Z(v)+Y(v),
     %%%%              74行中加号的前面部分对应Z(v)，加号的后面部分对应Y(v)
     for view_mark = 1:view_num
          data{view_mark} = anc_data{view_mark}*W*tq_l{view_mark}*kz_l{view_mark}+data_e{view_mark}*kz_e{view_mark};
     end
     %%%%
     
   %%%%更新相似度矩阵(需要一列一列地计算，每列对应一个样本的相似度)
   for data_mark = 1:data_num
       temp_G = 0;
       for view_mark = 1:view_num
          temp_G = temp_G + (data{view_mark}(:,data_mark)*ones(1,anc_num)-anc_data{view_mark})'*(data{view_mark}(:,data_mark)*ones(1,anc_num)-anc_data{view_mark});
       end
%        W(:,data_mark) = inv(temp_G+alpha*eye(anc_num))*ones(anc_num,1)/(ones(1,anc_num)*inv(temp_G+alpha*eye(anc_num))*ones(anc_num,1));
       W(:,data_mark) = quadprog(2*(temp_G+alpha*eye(anc_num)),f,A,b,Aeq,beq,lb,ub);  %%使用二次规划求解
   end
   %%%% 
    
    for view_mark = 1:view_num
        J(it_mark) = J(it_mark)+trace((data{view_mark}-anc_data{view_mark}*W)*(data{view_mark}-anc_data{view_mark}*W)');
    end
    J(it_mark) = J(it_mark)+alpha*trace(W*W');
    if it_mark>=2 && abs((J(it_mark)-J(it_mark-1))/J(it_mark-1))< stoperr        %% 收敛则结束本次迭代
          break;
    end
end
%%

% %%%%使用特征值分解求解
% %%所有样本间的相似度矩阵
% %%
% S = W'*inv(diag(sum(W,2)))*W;
% %%
% 
% [vec,val] = eig(S);
% [sort_val,index] = sort(diag(val),'descend');
% sort_vec = vec(:,index);
% new_data = sort_vec(:,2:new_dim+1);
% % new_data = sort_vec(:,1:new_dim);
% %%%%

% %%%%使用奇异值分解求解
% [U,val,V] = svd(W'*diag(sum(W,2))^-0.5,'econ');
% [sort_val,index] = sort(diag(val^2),'descend');
% sort_vec = U(:,index);
% new_data = sort_vec(:,2:new_dim+1);
% % new_data = sort_vec(:,1:new_dim);
% %%%%

% %%%%使用快速特征值分解（通过分解小矩阵得到大矩阵的特征分解结果）求解
SS = diag(sum(W,2))^-0.5*W*W'*diag(sum(W,2))^-0.5;
[vec,val] = eig(SS);
[sort_val,index] = sort(diag(val),'descend');
sort_vec = vec(:,index);
% need_val = diag(sort_val(2:new_dim+1));
% need_vec = sort_vec(:,2:new_dim+1);
need_val = diag(sort_val(1:new_dim));
need_vec = sort_vec(:,1:new_dim);
new_data = W'*diag(sum(W,2))^-0.5*need_vec*(need_val^-0.5);
% new_data = sort_vec(:,1:new_dim);
% %%%%


 %%%%把每一样本新表示的2范数化为1
    norm_new_data = repmat(sqrt(sum(new_data.*new_data,2)),1,size(new_data,2));
    %%avoid divide by zero
   norm_new_data = max(norm_new_data,1e-10);
%   norm_new_data(norm_new_data==0) = 1;
   new_data = new_data./norm_new_data;
 %%%%

%  %%%%%嵌入矩阵每一行的2范数化为1
%     norm_new_data = repmat(sqrt(sum(new_data.*new_data,2)),1,size(new_data,2));
%     %%avoid divide by zero
%    norm_new_data = max(norm_new_data,1e-10);
%    new_data = new_data./norm_new_data;
%  %%%%%