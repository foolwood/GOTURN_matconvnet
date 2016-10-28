a = zeros(227,227,3,1000);
%%
tic
k = 1;
for i=1:50,
    parfor j =1:500
    a(:,:,:,i+j) = zeros(227,227,3); 
    end
end
toc


a = zeros(227,227,3,1000);

%%
tic
for i=1:500,
    for j =1:500
    a(:,:,:,i+j) = zeros(227,227,3,100); 
    end
end
toc