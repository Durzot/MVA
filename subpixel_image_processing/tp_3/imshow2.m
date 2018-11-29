function f=imshow2(u,varargin)
  [ny,nx] = size(u);
  f = figure();
  imshow(u,varargin{:});
  pause(0.001);
  set(gcf, "position", [200 200 nx ny])
  set(gca, "position", [0 0 1 1])
endfunction
