function a=pivot(A,r,c)
% (c) jskl UoM
%
% pivot matrix A at row r and column c
% for zero pivot item no operation
%
x=A(r,c);               % x = pivot item
if x ~= 0
 rmax=length(A(:,1));   % rmax = number of rows
 A(r,:)=A(r,:)/x;       % divide row r by x (1 in pivot place)
 for i=1:rmax
  if i~=r               % making zeros in pivot column (except row r)
   A(i,:)=A(i,:)-A(r,:)*A(i,c);  % row i = row i - row r * A(i,c)
  end
 end
end
a=A;