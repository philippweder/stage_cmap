clear all
b{1} = [-2 0 1/sqrt(2)]';
b{2} = [1 sqrt(3)  1/sqrt(2)]';
b{3} = [1 -sqrt(3) 1/sqrt(2)]';
b{4} = [0 0 -3/sqrt(2)]';

norm(b{1})
norm(b{2})
norm(b{3})
norm(b{4})

for i = 1:4
    for j = i+1:4
        kl = setdiff([1,2,3,4],[i,j]);
        k = kl(1); l = kl(2);
        signe = permutationparity([i,j,k,l]);
        signe = (-1)^signe
        [i,j,k,l]
        tmp = cross(b{k},b{l});
        
        M1(k,l) = -tmp(1);
        M2(k,l) = -tmp(2);
        M3(k,l) = -tmp(3);
        M1(l,k) =  tmp(1);
        M2(l,k) =  tmp(2);
        M3(l,k) =  tmp(3);
    end
end
M1/1.2247
M2/(1.2247*sqrt(3))
M3/(1.2247*2*sqrt(2))