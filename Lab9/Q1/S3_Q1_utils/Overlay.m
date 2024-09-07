function Overlay(f, mask)

    m = max(f(:));
    fr = f;
    fg = f + mask / max(mask(:)) * m/2;
    fb = f;
   
    imshow(reshape([fr fg fb],[size(f,1) size(f,2) 3])/m, []);
    
    drawnow;