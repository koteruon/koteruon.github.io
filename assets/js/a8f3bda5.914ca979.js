"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([[1613],{47066:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>u,contentTitle:()=>i,default:()=>l,frontMatter:()=>o,metadata:()=>c,toc:()=>d});var s=t(74848),r=t(28453);const o={slug:"/notes/kubernetes-basics"},i="Reference Book",c={id:"notes/kubernetes-basics/reference-book",title:"Reference Book",description:"\u672c\u7bc7\u662f\u5f9e\u300a\u5f9e\u7570\u4e16\u754c\u6b78\u4f86\u767c\u73fe\u53ea\u5269\u81ea\u5df1\u4e0d\u6703 Kubernetes\u300b\u9019\u672c\u66f8\u6240\u6574\u7406\u51fa\u4f86\u7684\u91cd\u9ede\u7b46\u8a18\u548c\u5fc3\u5f97\uff0c",source:"@site/docs/notes/kubernetes-basics/00-reference-book.mdx",sourceDirName:"notes/kubernetes-basics",slug:"/notes/kubernetes-basics",permalink:"/docs/notes/kubernetes-basics",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:0,frontMatter:{slug:"/notes/kubernetes-basics"},sidebar:"notesSidebar",previous:{title:"Introduction",permalink:"/docs/notes"},next:{title:"Container Introduction",permalink:"/docs/notes/kubernetes-basics/container-intro"}},u={},d=[{value:"\u80cc\u666f\u8207\u51fa\u767c\u9ede",id:"\u80cc\u666f\u8207\u51fa\u767c\u9ede",level:2},{value:"\u66f8\u4e2d\u7684\u6838\u5fc3\u6982\u5ff5\u8207\u5be6\u52d9\u61c9\u7528",id:"\u66f8\u4e2d\u7684\u6838\u5fc3\u6982\u5ff5\u8207\u5be6\u52d9\u61c9\u7528",level:2},{value:"\u5be6\u4f5c\u7d93\u9a57\u8207\u6311\u6230",id:"\u5be6\u4f5c\u7d93\u9a57\u8207\u6311\u6230",level:2},{value:"\u5c0f\u7d50",id:"\u5c0f\u7d50",level:2}];function a(e){const n={a:"a",h1:"h1",h2:"h2",header:"header",img:"img",p:"p",...(0,r.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.header,{children:(0,s.jsx)(n.h1,{id:"reference-book",children:"Reference Book"})}),"\n",(0,s.jsxs)(n.p,{children:["\u672c\u7bc7\u662f\u5f9e\u300a",(0,s.jsx)(n.a,{href:"https://ithelp.ithome.com.tw/articles/10287149",children:"\u5f9e\u7570\u4e16\u754c\u6b78\u4f86\u767c\u73fe\u53ea\u5269\u81ea\u5df1\u4e0d\u6703 Kubernetes"}),"\u300b\u9019\u672c\u66f8\u6240\u6574\u7406\u51fa\u4f86\u7684\u91cd\u9ede\u7b46\u8a18\u548c\u5fc3\u5f97\uff0c\n\u518d\u7d50\u5408\u81ea\u5df1\u7684\u5be6\u969b\u64cd\u4f5c\u904e\u7a0b\u4e2d\u7684\u9ad4\u6703\u8207\u6311\u6230\uff0c\u6574\u7406\u51fa\u4f86\u8207\u5927\u5bb6\u5206\u4eab\u3002"]}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{alt:"book_cover",src:t(6814).A+"",width:"280",height:"413"})}),"\n",(0,s.jsx)(n.h2,{id:"\u80cc\u666f\u8207\u51fa\u767c\u9ede",children:"\u80cc\u666f\u8207\u51fa\u767c\u9ede"}),"\n",(0,s.jsx)(n.p,{children:"\u96a8\u8457\u96f2\u7aef\u6280\u8853\u7684\u767c\u5c55\uff0c\u5bb9\u5668\u5316\u6280\u8853\u6210\u70ba\u7576\u524d\u6700\u53d7\u95dc\u6ce8\u7684\u9818\u57df\u4e4b\u4e00\u3002\nKubernetes\uff0c\u4f5c\u70ba\u4e00\u500b\u958b\u6e90\u7684\u5bb9\u5668\u7ba1\u7406\u5e73\u53f0\uff0c\u63d0\u4f9b\u4e86\u9ad8\u6548\u7684\u5bb9\u5668\u7de8\u6392\u548c\u81ea\u52d5\u5316\u90e8\u7f72\u80fd\u529b\uff0c\u8b93\u8a31\u591a\u4f01\u696d\u5f97\u4ee5\u5be6\u73fe\u66f4\u9748\u6d3b\u7684\u8cc7\u6e90\u914d\u7f6e\u548c\u904b\u7dad\u7ba1\u7406\u3002\n\u7136\u800c\uff0c\u5c0d\u65bc\u525b\u63a5\u89f8\u9019\u9805\u6280\u8853\u7684\u4eba\u4f86\u8aaa\uff0c\u5b78\u7fd2 Kubernetes \u5f80\u5f80\u4f34\u96a8\u8457\u8a31\u591a\u6311\u6230\uff0c\u6211\u4e5f\u662f\u5176\u4e2d\u4e00\u4f4d\u3002\n\u56e0\u6b64\u5206\u4eab\u9019\u672c\u66f8\u6240\u5b78\u7684\u77e5\u8b58\uff0c\u9019\u6a23\u4e0d\u50c5\u6613\u65bc\u5438\u6536\u77e5\u8b58\uff0c\u4e5f\u589e\u52a0\u4e86\u5b78\u7fd2\u904e\u7a0b\u4e2d\u7684\u8da3\u5473\u6027\u3002"}),"\n",(0,s.jsx)(n.h2,{id:"\u66f8\u4e2d\u7684\u6838\u5fc3\u6982\u5ff5\u8207\u5be6\u52d9\u61c9\u7528",children:"\u66f8\u4e2d\u7684\u6838\u5fc3\u6982\u5ff5\u8207\u5be6\u52d9\u61c9\u7528"}),"\n",(0,s.jsx)(n.p,{children:"\u5728\u66f8\u4e2d\u7684\u7b46\u8a18\u90e8\u5206\uff0c\u4f5c\u8005\u8a73\u7d30\u4ecb\u7d39\u4e86 Kubernetes \u7684\u57fa\u672c\u6982\u5ff5\uff0c\u5305\u62ec Pod\u3001Node\u3001Cluster\u3001Service \u7b49\u6838\u5fc3\u7d44\u4ef6\u3002\n\u4e26\u642d\u914d\u8a31\u591a\u7bc4\u4f8b\u8b93\u8b80\u8005\u80fd\u5920\u8f15\u9b06\u4f7f\u7528\uff0c\u800c\u4e14\u4e0d\u9700\u8981\u81ea\u5df1\u67b6\u8a2d Kubernetes\uff0c\u5c31\u5f88\u5927\u7a0b\u5ea6\u7684\u5e6b\u52a9\u4f7f\u7528\u8005\u5165\u9580\u3002\n\u6b64\u5916\uff0c\u66f8\u4e2d\u4e5f\u6db5\u84cb\u4e86\u5982\u4f55\u4f7f\u7528 Kubernetes \u9032\u884c\u5bb9\u5668\u7684\u81ea\u52d5\u64f4\u5c55\u3001\u6efe\u52d5\u66f4\u65b0\u3001\u8cc7\u6e90\u7ba1\u7406\u7b49\u64cd\u4f5c\u3002\n\u9019\u4e9b\u90fd\u662f\u5be6\u969b\u5de5\u4f5c\u4e2d\u7d93\u5e38\u6703\u9047\u5230\u7684\u9700\u6c42\uff0c\u80fd\u5920\u5e6b\u52a9\u8b80\u8005\u5728\u5b78\u7fd2\u904e\u7a0b\u4e2d\u4e0d\u50c5\u638c\u63e1\u7406\u8ad6\u77e5\u8b58\uff0c\u9084\u80fd\u5920\u5be6\u969b\u61c9\u7528\u65bc\u5de5\u4f5c\u4e2d\u7684\u5834\u666f\u3002"}),"\n",(0,s.jsx)(n.h2,{id:"\u5be6\u4f5c\u7d93\u9a57\u8207\u6311\u6230",children:"\u5be6\u4f5c\u7d93\u9a57\u8207\u6311\u6230"}),"\n",(0,s.jsx)(n.p,{children:"\u5728\u5b78\u7fd2 Kubernetes \u7684\u904e\u7a0b\u4e2d\uff0c\u6211\u4e5f\u9047\u5230\u4e86\u4e00\u4e9b\u5be6\u969b\u7684\u6311\u6230\u3002\n\u4f8b\u5982\uff0c\u914d\u7f6e Kubernetes \u96c6\u7fa4\u6642\uff0c\u5982\u4f55\u6709\u6548\u5730\u7ba1\u7406\u96c6\u7fa4\u5167\u7684\u8cc7\u6e90\uff0c\u5982\u4f55\u907f\u514d\u4e0d\u540c Pod \u9593\u7684\u885d\u7a81\uff0c\u5982\u4f55\u8a2d\u5b9a\u5408\u9069\u7684\u670d\u52d9\u66b4\u9732\u65b9\u5f0f\u7b49\uff0c\u90fd\u662f\u9700\u8981\u6df1\u5165\u5b78\u7fd2\u7684\u8ab2\u984c\u3002\n\u9019\u4e9b\u6311\u6230\u5728\u5be6\u969b\u64cd\u4f5c\u4e2d\u8b93\u6211\u5c0d Kubernetes \u6709\u4e86\u66f4\u6df1\u523b\u7684\u7406\u89e3\uff0c\u4e26\u4e14\u80fd\u5920\u5728\u9762\u5c0d\u554f\u984c\u6642\u5feb\u901f\u5b9a\u4f4d\u548c\u89e3\u6c7a\u3002"}),"\n",(0,s.jsx)(n.h2,{id:"\u5c0f\u7d50",children:"\u5c0f\u7d50"}),"\n",(0,s.jsx)(n.p,{children:"\u7e3d\u7684\u4f86\u8aaa\uff0c\u300a\u5f9e\u7570\u4e16\u754c\u6b78\u4f86\u767c\u73fe\u53ea\u5269\u81ea\u5df1\u4e0d\u6703 Kubernetes\u300b\u9019\u672c\u66f8\u4ee5\u5176\u7368\u7279\u7684\u65b9\u5f0f\u5c07 Kubernetes \u7684\u5b78\u7fd2\u904e\u7a0b\u5448\u73fe\u7d66\u8b80\u8005\uff0c\u8b93\u4eba\u65e2\u80fd\u7372\u5f97\u77e5\u8b58\uff0c\u53c8\u80fd\u4eab\u53d7\u5b78\u7fd2\u7684\u904e\u7a0b\u3002\n\u7d50\u5408\u5be6\u969b\u7684\u64cd\u4f5c\u7d93\u9a57\uff0c\u9019\u672c\u66f8\u63d0\u4f9b\u4e86\u4e00\u500b\u6df1\u5165\u7406\u89e3 Kubernetes \u7684\u6709\u529b\u6307\u5357\u3002\n\u7121\u8ad6\u662f\u521d\u5b78\u8005\u9084\u662f\u6709\u4e00\u5b9a\u7d93\u9a57\u7684\u6280\u8853\u4eba\u54e1\uff0c\u90fd\u80fd\u5f9e\u4e2d\u7372\u5f97\u5bf6\u8cb4\u7684\u77e5\u8b58\u548c\u555f\u767c\u3002"})]})}function l(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(a,{...e})}):a(e)}},6814:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/book_cover-468822985431e32fc22ae44fc9f5e4b4.jpg"},28453:(e,n,t)=>{t.d(n,{R:()=>i,x:()=>c});var s=t(96540);const r={},o=s.createContext(r);function i(e){const n=s.useContext(o);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:i(e.components),s.createElement(o.Provider,{value:n},e.children)}}}]);