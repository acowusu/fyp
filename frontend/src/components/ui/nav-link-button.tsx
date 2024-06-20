import { buttonVariants } from "@/components/ui/button"
import { NavLink, NavLinkProps } from "react-router-dom"
import { ButtonProps } from "@/components/ui/button"
import React from "react"


interface NavLinkButtonProps {
    isActiveClass: string,
    isPendingClass: string,
}
export const NavLinkButton = React.forwardRef<HTMLAnchorElement, ButtonProps & NavLinkProps & NavLinkButtonProps>(({ children, variant, to, className, size, isActiveClass, isPendingClass, "aria-label": ariaLabel }, ref) => {
    const resolvedClasses = ({ isActive, isPending }: { isActive: boolean, isPending: boolean }) => buttonVariants({ variant, size }) + " " + className+ " " +( isPending ? isPendingClass : isActive ? isActiveClass : "")
    console.log("NavLinkButton", resolvedClasses({ isActive: true, isPending: true }))
    
    return (<NavLink ref={ref}
        to={to} aria-label={ariaLabel} className={resolvedClasses}>
        {children}
    </NavLink>
    )
})



// to="/dashboard"
//   className={({ isActive, isPending }) =>
//     isPending ? "rounded-lg" : isActive ? "rounded-lg" : "rounded-lg"
//   }
//   size="icon"
//   className="rounded-lg"
//   aria-label="Playground"