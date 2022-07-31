%% ����ê(anchor)�Ĳ���ȫ���ӽ��׾���
%% ���ý�����������������ê��֮������ƶȾ���W��Ҫ��W��Ԫ�ؾ�����0,ʹ�ö��ι滮���
%% maxiterΪ����������
%% stoperrΪ�жϵ������������
%% alphaΪ���򻯲���
%% JΪ���������Ŀ�꺯��ֵ
%% los_markΪÿ���ӽ���ȱʧ���������ı��
%% data_eΪ��������������ori_W�ǳ�ʼ��W,data_numΪ�ܵ�����������������ȱʧ������
%% alpha���趨�м���
%% �����в���ͳһ�ڽṹ��options��
%% ���Ż�Ŀ�꣺ min{Xv,W}SUMv=1~V||Xv-Av*W||^2+alpha||W||^2  s.t.Xv*SEv = XEv, W'1=1,  W>=0(1ΪԪ�ؾ�Ϊ1��������)
%%
function [data,W,new_data,J] = AIMSC(data_e,anc_data,ori_W,los_mark,options)

alpha = options.alpha;
maxiter = options.maxiter;
stoperr = options.stoperr;
new_dim = options.new_dim;

view_num = length(data_e);
[anc_num,data_num] = size(ori_W); %%anc_numΪê��ĸ���
dim = zeros(1,view_num); %��¼ÿ���ӽ�ά��������
tq_l = cell(1,view_num);  %ȱʧ��ȡ���� 
% tq_e = cell(1,view_num);  %������ȡ����
kz_l = cell(1,view_num); %ȱʧ��չ����
kz_e = cell(1,view_num); %������չ����
% W = cell(1,view_num); %ÿ���ӽ�������ê�������ƾ���
tot_mark = 1:data_num;   %�����ݱ�ǩ
% data_e = cell(1,view_num); %ÿ���ӽǵĴ�����������
% data_l = cell(1,view_num);  %ÿ���ӽǵ�ȱʧ��������
data = cell(1,view_num);  %ÿ���ӽǵ�������������

sum_dim = 0;

%% ���ι滮���������
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
    los_mark{view_mark} = sort(los_mark{view_mark});  %��ȱʧ�����ı�ǩ��˳������
%      tq_e{view_mark} = eye(data_num);
%      tq_e{view_mark}(:,los_mark{view_mark}) = [];
     ext_mark = setdiff(tot_mark,los_mark{view_mark});
     tq_l{view_mark} = eye(data_num);
     tq_l{view_mark}(:,ext_mark) = [];
     l_num = length(los_mark{view_mark}); %ÿ���ӽ�ȱʧ�����ĸ���
     e_num = data_num - l_num;  %ÿ���ӽǴ��������ĸ���
     kz_l{view_mark} = zeros(l_num,data_num);
     kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
     kz_e{view_mark} = zeros(e_num,data_num);
     kz_e{view_mark}(:,ext_mark) = eye(e_num);
%     data_e{view_mark} = data_t{view_mark}*tq_e{view_mark};
end
alpha = alpha*sum_dim/anc_num;

%%%%�����ƶȾ�����Ϊ��ʼֵ
W = ori_W;  
%%%%

%% ����������ƶȾ���W��������������data
J = zeros(maxiter,1);
for it_mark = 1:maxiter   %%�������
     %%%%����������������      �˴���Ӧ�������е� X(v)=Z(v)+Y(v),
     %%%%              74���мӺŵ�ǰ�沿�ֶ�ӦZ(v)���Ӻŵĺ��沿�ֶ�ӦY(v)
     for view_mark = 1:view_num
          data{view_mark} = anc_data{view_mark}*W*tq_l{view_mark}*kz_l{view_mark}+data_e{view_mark}*kz_e{view_mark};
     end
     %%%%
     
   %%%%�������ƶȾ���(��Ҫһ��һ�еؼ��㣬ÿ�ж�Ӧһ�����������ƶ�)
   for data_mark = 1:data_num
       temp_G = 0;
       for view_mark = 1:view_num
          temp_G = temp_G + (data{view_mark}(:,data_mark)*ones(1,anc_num)-anc_data{view_mark})'*(data{view_mark}(:,data_mark)*ones(1,anc_num)-anc_data{view_mark});
       end
%        W(:,data_mark) = inv(temp_G+alpha*eye(anc_num))*ones(anc_num,1)/(ones(1,anc_num)*inv(temp_G+alpha*eye(anc_num))*ones(anc_num,1));
       W(:,data_mark) = quadprog(2*(temp_G+alpha*eye(anc_num)),f,A,b,Aeq,beq,lb,ub);  %%ʹ�ö��ι滮���
   end
   %%%% 
    
    for view_mark = 1:view_num
        J(it_mark) = J(it_mark)+trace((data{view_mark}-anc_data{view_mark}*W)*(data{view_mark}-anc_data{view_mark}*W)');
    end
    J(it_mark) = J(it_mark)+alpha*trace(W*W');
    if it_mark>=2 && abs((J(it_mark)-J(it_mark-1))/J(it_mark-1))< stoperr        %% ������������ε���
          break;
    end
end
%%

% %%%%ʹ������ֵ�ֽ����
% %%��������������ƶȾ���
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

% %%%%ʹ������ֵ�ֽ����
% [U,val,V] = svd(W'*diag(sum(W,2))^-0.5,'econ');
% [sort_val,index] = sort(diag(val^2),'descend');
% sort_vec = U(:,index);
% new_data = sort_vec(:,2:new_dim+1);
% % new_data = sort_vec(:,1:new_dim);
% %%%%

% %%%%ʹ�ÿ�������ֵ�ֽ⣨ͨ���ֽ�С����õ������������ֽ��������
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


 %%%%��ÿһ�����±�ʾ��2������Ϊ1
    norm_new_data = repmat(sqrt(sum(new_data.*new_data,2)),1,size(new_data,2));
    %%avoid divide by zero
   norm_new_data = max(norm_new_data,1e-10);
%   norm_new_data(norm_new_data==0) = 1;
   new_data = new_data./norm_new_data;
 %%%%

%  %%%%%Ƕ�����ÿһ�е�2������Ϊ1
%     norm_new_data = repmat(sqrt(sum(new_data.*new_data,2)),1,size(new_data,2));
%     %%avoid divide by zero
%    norm_new_data = max(norm_new_data,1e-10);
%    new_data = new_data./norm_new_data;
%  %%%%%