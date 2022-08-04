%% data: complete data matrix
%% W: similarity matrix between data points and anchor points 
%% new_data: new data obtained by AIMSC 
%% J: value of objective function
%% data_e: existing data matrix，anc_data: anchor data matrix, ori_W: original W, los_mark: index number of missing data
%% options is a struct combined by alpha, maxiter, stoperr and new_dim,   
%% alpha: regularization parameter, maxiter: the maximum number of iteration,  stoperr: threshold of iteration stop,  new_dim: dimension of new_data
%% objective function： min{Xv,W}SUMv=1~V||Xv-Av*W||^2+alpha||W||^2  s.t.Xv*SEv = XEv, W'1=1,  W>=0
%%

function [data,W,new_data,J] = AIMSC(data_e,anc_data,ori_W,los_mark,options)

alpha = options.alpha;
maxiter = options.maxiter;
stoperr = options.stoperr;
new_dim = options.new_dim;

view_num = length(data_e);
[anc_num,data_num] = size(ori_W); %% number of anchors
dim = zeros(1,view_num); 
tq_l = cell(1,view_num);  
kz_l = cell(1,view_num); 
kz_e = cell(1,view_num); 
tot_mark = 1:data_num;   
data = cell(1,view_num);  % complete data matrix of each view

sum_dim = 0;

%% parameters of quadratic programming
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
    los_mark{view_mark} = sort(los_mark{view_mark});  
     ext_mark = setdiff(tot_mark,los_mark{view_mark});
     tq_l{view_mark} = eye(data_num);
     tq_l{view_mark}(:,ext_mark) = [];
     l_num = length(los_mark{view_mark}); % number of missing data of each view
     e_num = data_num - l_num;  % number of existing data of each view
     kz_l{view_mark} = zeros(l_num,data_num);
     kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
     kz_e{view_mark} = zeros(e_num,data_num);
     kz_e{view_mark}(:,ext_mark) = eye(e_num);
end
alpha = alpha*sum_dim/anc_num;

%%%% initialize W
W = ori_W;  
%%%%

%% obtain W and data by iteration
J = zeros(maxiter,1);
for it_mark = 1:maxiter   
     %%%% update data
     for view_mark = 1:view_num
          data{view_mark} = anc_data{view_mark}*W*tq_l{view_mark}*kz_l{view_mark}+data_e{view_mark}*kz_e{view_mark};
     end
     %%%%
     
   %%%% update W
   for data_mark = 1:data_num
       temp_G = 0;
       for view_mark = 1:view_num
          temp_G = temp_G + (data{view_mark}(:,data_mark)*ones(1,anc_num)-anc_data{view_mark})'*(data{view_mark}(:,data_mark)*ones(1,anc_num)-anc_data{view_mark});
       end
       W(:,data_mark) = quadprog(2*(temp_G+alpha*eye(anc_num)),f,A,b,Aeq,beq,lb,ub);  %% sovle W by quadratic programming
   end
   %%%% 
    
    for view_mark = 1:view_num
        J(it_mark) = J(it_mark)+trace((data{view_mark}-anc_data{view_mark}*W)*(data{view_mark}-anc_data{view_mark}*W)');
    end
    J(it_mark) = J(it_mark)+alpha*trace(W*W');
    if it_mark>=2 && abs((J(it_mark)-J(it_mark-1))/J(it_mark-1))< stoperr       
          break;
    end
end
%%


% %%%% fast eigenvalue decompsition
SS = diag(sum(W,2))^-0.5*W*W'*diag(sum(W,2))^-0.5;
[vec,val] = eig(SS);
[sort_val,index] = sort(diag(val),'descend');
sort_vec = vec(:,index);
need_val = diag(sort_val(1:new_dim));
need_vec = sort_vec(:,1:new_dim);
new_data = W'*diag(sum(W,2))^-0.5*need_vec*(need_val^-0.5);
% %%%%

 norm_new_data = repmat(sqrt(sum(new_data.*new_data,2)),1,size(new_data,2));
 norm_new_data = max(norm_new_data,1e-10);   %% avoid divide by zero
 new_data = new_data./norm_new_data;

